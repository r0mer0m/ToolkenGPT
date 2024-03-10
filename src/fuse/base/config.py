from transformers import AutoConfig

class AugmentedConfig(AutoConfig):
    
    def __init__(self, augment_embeddings=False, aug_vocab_size=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.augment_embeddings = augment_embeddings
        self.aug_vocab_size = aug_vocab_size
        
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, augment_embeddings=False, aug_vocab_size=0, **kwargs):
        config = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        config.augment_embeddings = augment_embeddings
        config.aug_vocab_size = aug_vocab_size
        return config
        