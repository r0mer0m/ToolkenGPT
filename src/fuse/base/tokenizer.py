# from sentencepiece import SentencePieceProcessor
from logging import getLogger
import warnings
import json
from transformers import LlamaTokenizer

logger = getLogger()


class AugmentedTokenizer(LlamaTokenizer):
    SYMBOL_SYNTHAX = '<<<{api_name}>>>'
    
    def __init__(self, augmentation_config: dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.augmentation_config = augmentation_config
        if augmentation_config and augmentation_config.api_names:
            
            self.base_vocab_size = self.vocab_size
            
            # Option 1: no control tokens
            # self.n_control_tokens = 0
            
            
            ### Option 2: control tokens ['EOC', 'EOR']
            control_tokens = ['EOC', 'EOR']
            
            # Add control tokens
            self.api_control_tokens = [self.SYMBOL_SYNTHAX.format(api_name=control_token) for control_token in control_tokens]
            self.n_control_tokens = self.add_tokens(self.api_control_tokens, special_tokens=True)
            
            (self.eoc_id,
            self.eor_id) = self.convert_tokens_to_ids(self.api_control_tokens)
            
            assert self.n_control_tokens == len(self.api_control_tokens), "Your tokenizer contains one or more api_control_tokens by default. Please update the `SYMBOL_SYNTHAX` \
                        variable in `AugmentedTokenizer` to make it unique"
            
            
            ### Option 3: control tokens ['BOC', 'EOC', 'BOR', 'EOR']
            # control_tokens = ['BOC', 'EOC', 'BOR', 'EOR']
            
            # # Add control tokens
            # self.api_control_tokens = [self.SYMBOL_SYNTHAX.format(api_name=control_token) for control_token in control_tokens]
            # self.n_control_tokens = self.add_tokens(self.api_control_tokens, special_tokens=True)
            
            # (self.boc_id, 
            # self.eoc_id, 
            # self.bor_id, 
            # self.eor_id) = self.convert_tokens_to_ids(self.api_control_tokens)
            
            # assert self.n_control_tokens == len(self.api_control_tokens), "Your tokenizer contains one or more api_control_tokens by default. Please update the `SYMBOL_SYNTHAX` \
            #             variable in `AugmentedTokenizer` to make it unique"
            
            
            
            ####### Add api tokens
            self.api_names = augmentation_config.api_names
            
            # self.api_names = ["add", "subtract", "multiply", "divide", "power", "sqrt", "log", "ln", "lcm", "gcd", "remainder", "choose", "permutate"]
            self.api_symbols = [self.SYMBOL_SYNTHAX.format(api_name=api_name) for api_name in self.api_names]
            self.n_api_tokens = self.add_tokens(self.api_symbols, special_tokens=True)
            self.api_ids = self.convert_tokens_to_ids(self.api_symbols)
            
            self.id_to_api = dict(zip(self.api_ids, self.api_names))
            self.api2ids = dict(zip(self.api_names, self.api_ids))
            self.symbol_to_api = dict(zip(self.api_symbols, self.api_names))
            self.symbol_to_id = dict(zip(self.api_symbols, self.api_ids))
            
            self.name2tokenized_defition = {
                _id: self.encode(augmentation_config.api_definitions[api_name])[1:]
                for _id, api_name in self.id_to_api.items()
            }
            
            assert self.n_api_tokens == len(self.api_names), "Your tokenizer contains one or more api_symbols by default. Please update the `SYMBOL_SYNTHAX` \
                        variable in `AugmentedTokenizer` to make it unique"
            
            self.aug_vocab_size = self.n_control_tokens + self.n_api_tokens
            
            assert len(self) == self.base_vocab_size + self.aug_vocab_size, f"{self.vocab_size} != {len(self)}"
            
        else:
            warnings.warn('`augmentation_config` nor provided of provided `api_names` is empty. Resorting to default tokenizer.')
            self.aug_vocab_size = 0
            self.n_api_tokens = 0
        
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path:str, augmentation_config:str='', *args, **kwargs):
        if not augmentation_config:
            warnings.warn('`augmentation_config_path` nor provided.')
            augmentation_config = {}

        return super().from_pretrained(pretrained_model_name_or_path, augmentation_config=augmentation_config, *args, **kwargs)
