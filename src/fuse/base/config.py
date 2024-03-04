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
        return config
        