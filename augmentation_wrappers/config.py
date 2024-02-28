from transformers import AutoConfig

class AugmentedConfig(AutoConfig):
    
    def __init__(self, augment=False, aug_vocab_size=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.aug_vocab_size = aug_vocab_size
        self.augment = augment
        
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, augment=False, aug_vocab_size=0, **kwargs):
        config = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        config.aug_vocab_size = aug_vocab_size
        config.augment = augment
        # config.max_batch_size = 32
        # config.max_seq_len = 2048
        # config.norm_eps = 1e-5
        # config.multiple_of = 256
        return config
        