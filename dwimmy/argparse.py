"""Drop-in replacement for argparse.ArgumentParser with DWIM capabilities"""

import argparse
import sys
from typing import List, Dict, Optional, Any
from dwimmy.core.matcher import SemanticMatcher
from dwimmy.core.utils import is_interactive, load_config


class ArgumentParser(argparse.ArgumentParser):
    """ArgumentParser with semantic error recovery"""
    
    def __init__(self, *args, **kwargs):
        # Extract dwimmy-specific options
        self._dwim_enabled = kwargs.pop('dwim_enabled', True)
        self._dwim_threshold = kwargs.pop('dwim_threshold', 0.75)
        
        super().__init__(*args, **kwargs)
        
        self._matcher: Optional[SemanticMatcher] = None
        self._search_space: Optional[Dict] = None
        self._config = load_config()
        
        # Check if disabled via config
        if not self._config['enabled']:
            self._dwim_enabled = False
    
    def _build_search_space(self) -> Dict[str, Any]:
        """Extract all searchable components from the parser"""
        search_space = {
            'flags': [],
            'flag_metadata': {},
            'choices': {},
            'subcommands': [],
            'flag_to_dest': {},
        }
        
        # Iterate through all registered actions
        for action in self._actions:
            # Extract flags (option_strings like ['--config', '-c'])
            if action.option_strings:
                search_space['flags'].extend(action.option_strings)
                
                # Store metadata for each flag
                for opt_string in action.option_strings:
                    search_space['flag_to_dest'][opt_string] = action.dest
                
                search_space['flag_metadata'][action.dest] = {
                    'help': action.help or '',
                    'type': action.type,
                    'choices': action.choices,
                    'required': getattr(action, 'required', False),
                }
            
            # Extract choices (enum values)
            if action.choices:
                search_space['choices'][action.dest] = list(action.choices)
            
            # Handle subparsers
            if isinstance(action, argparse._SubParsersAction):
                search_space['subcommands'] = list(action.choices.keys())
        
        return search_space
    
    def _get_flag_dest(self, flag: str) -> Optional[str]:
        """Get the destination name for a flag"""
        if self._search_space:
            return self._search_space['flag_to_dest'].get(flag)
        return None
    
    def _reconstruct_command(self, tokens: List[str]) -> Optional[List[str]]:
        """Attempt to reconstruct valid command from tokens"""
        if not self._search_space:
            return None
        
        reconstructed = []
        i = 0
        changed = False  # Track if we made any changes
        
        # Exclude help flags from matching candidates
        searchable_flags = [f for f in self._search_space['flags'] 
                           if f not in ['-h', '--help', '--version']]
        
        while i < len(tokens):
            token = tokens[i]
            matched = False
            
            # Case 1: Token starts with dash - likely a malformed flag
            if token.startswith('-'):
                result = self._matcher.find_closest(
                    token,
                    searchable_flags,
                    threshold=0.50,
                    exclude=['-h', '--help']
                )
                if result:
                    matched_flag, score = result
                    if matched_flag != token:  # Only count if different
                        changed = True
                    reconstructed.append(matched_flag)
                    matched = True
                    print(f"  Matched '{token}' → '{matched_flag}' (confidence: {score:.2f})", 
                          file=sys.stderr)
            
            # Case 2: Token doesn't start with dash - check if it should be a flag
            elif not matched:
                # Try matching against flag names without dashes
                flag_names = [f.lstrip('-') for f in searchable_flags]
                result = self._matcher.find_closest(
                    token,
                    flag_names,
                    threshold=0.60  # Slightly higher for missing-dash case
                )
                if result:
                    matched_name, score = result
                    # Find original flag with dashes
                    for flag in searchable_flags:
                        if flag.lstrip('-') == matched_name:
                            reconstructed.append(flag)
                            matched = True
                            changed = True
                            print(f"  Matched '{token}' → '{flag}' (confidence: {score:.2f})", 
                                  file=sys.stderr)
                            break
            
            # Case 3: Check if it's a choice value for previous flag
            if not matched and reconstructed and reconstructed[-1].startswith('-'):
                flag_dest = self._get_flag_dest(reconstructed[-1])
                if flag_dest and flag_dest in self._search_space['choices']:
                    result = self._matcher.find_closest(
                        token,
                        self._search_space['choices'][flag_dest],
                        threshold=0.45
                    )
                    if result:
                        matched_choice, score = result
                        if matched_choice != token:
                            changed = True
                        reconstructed.append(matched_choice)
                        matched = True
                        print(f"  Matched '{token}' → '{matched_choice}' (confidence: {score:.2f})", 
                              file=sys.stderr)
            
            # Case 4: Pass through as-is (positional arg, value, etc.)
            if not matched:
                reconstructed.append(token)
            
            i += 1
        
        # Return None only if no changes were made
        return reconstructed if changed else None    

    def _prompt_and_retry(self, suggestion: List[str]) -> Any:
        """Show suggestion and retry parsing if accepted"""
        suggested_cmd = ' '.join(suggestion)
        
        # Check if we're in interactive mode
        interactive = self._config['interactive']
        if interactive == 'never' or (interactive == 'auto' and not is_interactive()):
            print(f"\nSuggestion: {suggested_cmd}", file=sys.stderr)
            print("Run with DWIMMY_INTERACTIVE=always to approve interactively", 
                  file=sys.stderr)
            return None
        
        # Auto-correct mode (risky!)
        if self._config['auto_correct']:
            print(f"\nAuto-correcting to: {suggested_cmd}", file=sys.stderr)
            sys.argv[1:] = suggestion
            return super().parse_args()
        
        # Interactive prompt
        print(f"\nDid you mean: {suggested_cmd}", file=sys.stderr)
        try:
            response = input("Run this command? [y/n]: ")
        except (EOFError, KeyboardInterrupt):
            print("\nAborted", file=sys.stderr)
            return None
        
        if response.lower() in ['y', 'yes']:
            sys.argv[1:] = suggestion
            return super().parse_args()
        else:
            print("Aborted", file=sys.stderr)
            return None
    
    def _attempt_dwim_recovery(self, error_message: str) -> Optional[List[str]]:
        """Try to recover from parsing error using semantic matching"""
        # Initialize matcher if needed
        if self._matcher is None:
            from dwimmy.core.utils import SimpleSpinner, is_interactive
            
            spinner = None
            if is_interactive():
                spinner = SimpleSpinner(
                    text='Interpreting your command',
                    spinner='dots',
                    stream=sys.stderr
                )
                spinner.start()
            
            try:
                self._matcher = SemanticMatcher()
                self._search_space = self._build_search_space()
                
                # Try to load pre-computed embeddings from package
                self._matcher._load_embeddings_from_package()
                
                if spinner:
                    spinner.succeed('DWIM ready to interpret')
                
                # Embed with caching - uses package embeddings if available
                self._matcher.embed_search_space(
                    self._search_space, 
                    show_spinner=False,
                    use_cache=True,
                    save_cache=False  # Don't save when using package embeddings
                )
            except Exception as e:
                if spinner:
                    spinner.fail(f'Failed to initialize DWIM: {e}')
                raise
        
        # Get raw user input
        user_argv = sys.argv[1:]
        
        if not user_argv:
            return None
        
        print("\nAttempting to interpret your command...", file=sys.stderr)
        
        # Try to reconstruct valid command
        suggestion = self._reconstruct_command(user_argv)
        
        return suggestion

        def error(self, message: str):

            """Override error to add DWIM recovery"""

            if not self._dwim_enabled:

                super().error(message)

                return

            

            # Try DWIM recovery

            try:

                suggestion = self._attempt_dwim_recovery(message)

                

                if suggestion:

                    result = self._prompt_and_retry(suggestion)

                    if result is not None:

                        return result

            except Exception as e:

                print(f"DWIM recovery failed: {e}", file=sys.stderr)

            

            # Fall back to standard error

            super().error(message)

    
        

