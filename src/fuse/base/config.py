from transformers import AutoConfig

class AugmentedConfig(AutoConfig):
    
    def __init__(self, embedding_augmentation_type=False, aug_vocab_size=0, aug_token_bias=0, initialization='random', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding_augmentation_type = embedding_augmentation_type
        self.aug_vocab_size = aug_vocab_size
        self.aug_token_bias = aug_token_bias
        self.initialization = initialization
        
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, 
                        embedding_augmentation_type=False, 
                        aug_vocab_size=0, 
                        aug_token_bias=0, 
                        initialization='random', 
                        **kwargs):
        config = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        config.embedding_augmentation_type = embedding_augmentation_type
        config.aug_vocab_size = aug_vocab_size
        config.aug_token_bias = aug_token_bias
        config.initialization=initialization # Supported options: random, semantic
        return config
        