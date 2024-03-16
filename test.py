from src.fuse.base.tokenizer import AugmentedTokenizer
from omegaconf import OmegaConf

augmentation_config = OmegaConf.load('./configs/runs/augmentation/funcqa_specific.yaml')
tokenizer = AugmentedTokenizer.from_pretrained(
        'meta-llama/Llama-2-7b-chat-hf',
        augmentation_config=augmentation_config,
        )
tokenizer.pad_token_id = tokenizer.eos_token_id


print(tokenizer.encode('will be left unpacked?\nA: The number of apples left unpacked'))
print(tokenizer.convert_ids_to_tokens(tokenizer.encode('A:')))
print(tokenizer.encode('A:'))
print(tokenizer.convert_ids_to_tokens(tokenizer.encode('\nA:')))
print(tokenizer.encode('\nA:'))
