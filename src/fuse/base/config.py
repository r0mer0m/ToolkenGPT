from transformers import AutoConfig

class AugmentedConfig(AutoConfig):
    
    def __init__(self, embedding_augmentation_type=False, aug_vocab_size=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding_augmentation_type = embedding_augmentation_type
        self.aug_vocab_size = aug_vocab_size
        
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, embedding_augmentation_type=False, aug_vocab_size=0, **kwargs):
        config = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        config.embedding_augmentation_type = embedding_augmentation_type
        config.aug_vocab_size = aug_vocab_size
        return config
        