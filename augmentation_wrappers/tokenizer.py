# from sentencepiece import SentencePieceProcessor
from logging import getLogger
import warnings
import json
from transformers import LlamaTokenizer

logger = getLogger()


class AugmentedTokenizer(LlamaTokenizer):
    SYMBOL_SYNTHAX = '<{api_name}>'
    
    def __init__(self, augmentation_config: dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        if augmentation_config and augmentation_config['api_names']:
            # special tokens
            self.base_vocab_size = self.vocab_size
            
            # Add control tokens
            self.api_control_tokens = [self.SYMBOL_SYNTHAX.format(api_name=control_token) for control_token in ['BOC', 'EOC', 'BOR', 'EOR']]
            self.n_control_tokens = self.add_tokens(self.api_control_tokens, special_tokens=True)
            
            (self.boc_id, 
            self.eoc_id, 
            self.bor_id, 
            self.eor_id) = self.convert_tokens_to_ids(self.api_control_tokens)
            
            assert self.n_control_tokens == len(self.api_control_tokens), "Your tokenizer contains one or more api_control_tokens by default. Please update the `SYMBOL_SYNTHAX` \
                        variable in `AugmentedTokenizer` to make it unique"
            
            # Add api tokens
            self.api_names = augmentation_config['api_names']
            
            # self.api_names = ["add", "subtract", "multiply", "divide", "power", "sqrt", "log", "ln", "lcm", "gcd", "remainder", "choose", "permutate"]
            self.api_symbols = [self.SYMBOL_SYNTHAX.format(api_name=api_name) for api_name in self.api_names]
            
            self.n_api_tokens = self.add_tokens(self.api_symbols, special_tokens=True)
            self.id_to_api = {
                idx: sym
                for idx, sym in zip(self.convert_tokens_to_ids(self.api_symbols), self.api_names)
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
    def from_pretrained(cls, pretrained_model_name_or_path:str, augmentation_config_path:str='', *args, **kwargs):
        if augmentation_config_path:
            with open(augmentation_config_path, 'r') as fp:
                augmentation_config = json.load(fp)
        else:
            warnings.warn('`augmentation_config_path` nor provided.')
            augmentation_config = {}

        return super().from_pretrained(pretrained_model_name_or_path, augmentation_config=augmentation_config, *args, **kwargs)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # @staticmethod
    # def _augment_tokenizer(cls, tokenizer, augmentation_config):
    #     # special tokens
    #     tokenizer.base_vocab_size = tokenizer.vocab_size
        
    #     # Add control tokens
    #     tokenizer.api_control_tokens = [self.SYMBOL_SYNTHAX.format(api_names=control_token) for control_token in ['BOC', 'EOC', 'BOR', 'EOR']]
    #     tokenizer.n_control_tokens = tokenizer.add_tokens(tokenizer.api_control_tokens, special_tokens=True)
        
    #     (tokenizer.boc_id, 
    #      tokenizer.eoc_id, 
    #      tokenizer.bor_id, 
    #      tokenizer.eor_id) = tokenizer.convert_tokens_to_ids(tokenizer.api_control_tokens)
        
    #     assert (tokenizer.n_api_tokens == len(tokenizer.api_names), 
    #             "Your tokenizer contains one or more api_control_tokens by default. Please update the `SYMBOL_SYNTHAX` \
    #                 variable in `AugmentedTokenizer` to make it unique")
        
    #     # Add api tokens
    #     tokenizer.api_names = augmentation_config['api_names']
        
    #     tokenizer.api_names = ["add", "subtract", "multiply", "divide", "power", "sqrt", "log", "ln", "lcm", "gcd", "remainder", "choose", "permutate"]
    #     tokenizer.api_symbols = [self.SYMBOL_SYNTHAX.format(api_name=api_name) for api_name in tokenizer.api_names]
        
    #     tokenizer.n_api_tokens = tokenizer.add_tokens(tokenizer.api_symbols, special_tokens=True)
    #     tokenizer.id_to_api = {
    #         idx: sym
    #         for idx, sym in zip(tokenizer.convert_tokens_to_ids(tokenizer.api_symbols), tokenizer.api_names)
    #     }
        
    #     assert (tokenizer.n_api_tokens == len(tokenizer.api_names), 
    #             "Your tokenizer contains one or more api_symbols by default. Please update the `SYMBOL_SYNTHAX` \
    #                 variable in `AugmentedTokenizer` to make it unique")
        
    #     tokenizer.aug_vocab_size = tokenizer.n_control_tokens + tokenizer.n_api_tokens
        
    #     assert tokenizer.vocab_size == tokenizer.base_vocab_size + tokenizer.aug_vocab_size


# class Tokenizer:
#     def __init__(self, model_path: str):
#         # reload tokenizer
#         assert os.path.isfile(model_path), model_path
#         self.sp_model = SentencePieceProcessor(model_file=model_path)
#         logger.info(f"Reloaded SentencePiece model from {model_path}")

#         # BOS / EOS token IDs
#         self.n_words: int = self.sp_model.vocab_size()
#         self.bos_id: int = self.sp_model.bos_id()
#         self.eos_id: int = self.sp_model.eos_id()
#         self.pad_id: int = self.sp_model.pad_id()
#         logger.info(
#             f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}"
#         )
#         assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

#     def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
#         assert type(s) is str
#         t = self.sp_model.encode(s)
#         if bos:
#             t = [self.bos_id] + t
#         if eos:
#             t = t + [self.eos_id]
#         return t

#     def decode(self, t: List[int]) -> str:
#         return self.sp_model.decode(t)


# class AugmentedTokenizer:
#     def __init__(self, tokenizer_dir: str):
#         # reload tokenizer
#         assert os.path.isdir(tokenizer_dir), tokenizer_dir
#         (self.sp_model, 
#          self.augmentation_config) = self._load(tokenizer_dir)
#         logger.info(f"Reloaded Augmented SentencePiece model from {tokenizer_dir}")

#         # BOS / EOS token IDs
#         self.n_base_words: int = self.augmentation_config['n_base_words']
#         self.n_aug_words: int = self.augmentation_config['n_aug_words']
#         self.mapping_offset = self.n_base_words - self.augmentation_config['insertion_index']
#         self.fun_map = {
#             (int(i) + self.augmentation_config['n_base_words'] - (self.augmentation_config['insertion_index'] - 1)): sym 
#             for i, sym in self.augmentation_config['fun_map'].items()}
        
#         self.n_fun: int = self.augmentation_config['n_fun']
#         self.boc_id: int = self.augmentation_config['boc_id'] + self.mapping_offset 
#         self.eoc_id: int = self.augmentation_config['eoc_id'] + self.mapping_offset 
#         self.bor_id: int = self.augmentation_config['bor_id'] + self.mapping_offset 
#         self.eor_id: int = self.augmentation_config['eor_id'] + self.mapping_offset 
        
#         self.bos_id: int = self.sp_model.bos_id()
#         self.eos_id: int = self.sp_model.eos_id()
#         self.pad_id: int = self.sp_model.pad_id()
#         logger.info(
#             f"#base words: {self.n_base_words} - #augmented words: {self.n_aug_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}"
#         )
#         assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

#     def _load(self, tokenizer_dir: str):
#         out_model_path = osp.join(tokenizer_dir, 'tokenizer.model')
#         out_config_path = osp.join(tokenizer_dir, 'tokenizer.config')
        
#         with open(out_config_path, 'r') as fp:
#             config = json.load(fp)
#         sp_model = SentencePieceProcessor(model_file=out_model_path)
        
#         return sp_model, config
    
#     def _encode_remap(self, t: List[int]) -> List[int]:
#         numpy_t = np.array(t)
#         new_tokens_start_idx = self.augmentation_config['insertion_index']
#         base_tokens_start_idx = self.augmentation_config['insertion_index'] + self.augmentation_config['n_aug_words']
        
#         # identify token types (base non-control and new)
#         mask_base_non_control = (
#             numpy_t >= base_tokens_start_idx
#         )
#         mask_new_tokens = (
#             (numpy_t >= new_tokens_start_idx) & 
#             (numpy_t < base_tokens_start_idx)
#         )
        
#         # re-map
#         numpy_t[mask_base_non_control] -= self.augmentation_config['n_aug_words']
#         numpy_t[mask_new_tokens] += self.mapping_offset 
        
#         return numpy_t.tolist()
    
#     def _decode_remap(self, t: List[int]) -> List[int]:
#         numpy_t = np.array(t)
#         # identify token types (base non-control and new)
#         new_tokens_start_idx = self.augmentation_config['n_base_words'] + 1
#         base_tokens_start_idx = self.augmentation_config['insertion_index']
        
#         mask_base_non_control = (
#             (numpy_t >= base_tokens_start_idx) &
#             (numpy_t < new_tokens_start_idx)
#         )
#         mask_new_tokens = (
#             numpy_t >= new_tokens_start_idx
#         )
        
#         # re-map
#         numpy_t[mask_base_non_control] += self.augmentation_config['n_aug_words']
#         numpy_t[mask_new_tokens] -= self.mapping_offset 
        
#         return numpy_t.tolist()
    
#     def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
#         assert type(s) is str
#         t = self.sp_model.encode(s)
#         if bos:
#             t = [self.bos_id] + t
#         if eos:
#             t = t + [self.eos_id]
#         t = self._encode_remap(t)
#         return t

#     def decode(self, t: List[int]) -> str:
#         t = self._decode_remap(t)
#         return self.sp_model.decode(t)
