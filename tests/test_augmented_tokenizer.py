"""
Run as 

```
python dev/dev.py
```

"""
import sys
sys.path.append("/home/karypisg/romer333/projects/LLM-tools/ToolkenGPT")

from llama.tokenizer import AugmentedTokenizer

augmented_tokenizer_dir = "/home/karypisg/romer333/projects/LLM-tools/ToolkenGPT/augmented_tokenizer/"
tokenizer = AugmentedTokenizer("/home/karypisg/romer333/projects/LLM-tools/ToolkenGPT/augmented_tokenizer/")


test = "This is a test with a <BOC> and a <EOC> and a <BOR> and a <EOR>."
print("Test in:\n\t", test)
ids = tokenizer.encode(test, bos=True, eos=True)
print("Ids:\n\t", ids)
print("Test out:\n\t", tokenizer.decode(ids))

for id in [32001,32002,32003,32004]:
    print(f"Id {id} to piece: {tokenizer.decode([id])}")

