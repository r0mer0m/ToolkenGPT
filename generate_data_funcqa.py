import re
import json
import torch
from llama_augmented.tokenizer import AugmentedTokenizer


def process_call_annotation(annotation):
    annotation = re.sub(r"=", "<EOC><BOR>", annotation)
    annotation = re.sub(r"\<eoe\>", "<EOR>", annotation)
    annotation = "<BOC>" + annotation
    return annotation

def main(data):
    
    tokenizer = AugmentedTokenizer('../ToolkenGPT/augmented_tokenizer')
    
    out_data = []
    for record in data:
        text = record['text']
        for tar_number, tar_eq in zip(record['tar_number'], record['tar_eq']):
            tar_eq = process_call_annotation(tar_eq)
            text = re.sub(tar_number, tar_eq + " " + tar_number, text)
        tokens = torch.Tensor(tokenizer.encode(text, bos=True, eos=True)).long()
        
        boc_ids = torch.nonzero(tokens==tokenizer.boc_id, as_tuple=True)[0].tolist()
        # eoc_ids = torch.nonzero(tokens==tokenizer.eoc_id, as_tuple=True)[0].tolist()
        bor_ids = torch.nonzero(tokens==tokenizer.bor_id, as_tuple=True)[0].tolist()
        eor_ids = torch.nonzero(tokens==tokenizer.eor_id, as_tuple=True)[0].tolist()
        
        tokens = tokens.tolist()
        
        out_data.append({
            'text': text, 
            'tar_eq': record['tar_eq'], 
            'tar_number': record['tar_number'],
            'fun_ids': [tokens[idx + 1] for idx in boc_ids],
            # 'eoc_ids': eoc_ids,
            'bor_ids': bor_ids,
            'eor_ids': eor_ids
            })
        # break
    return out_data
    

if __name__ == '__main__':
    with open('../data/funcqa/train.json', 'r') as fp:
        data = json.load(fp)

    out_data = main(data)

    with open('../augmentation_data/funcqa/train.json', 'w') as fp:
        json.dump(out_data, fp)
