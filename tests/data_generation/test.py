from pprint import pprint
import json
from funcqa import process_record, get_tokenizer#, process_call

tokenizer = get_tokenizer()

with open('../data/funcqa/train.json', 'r') as fp:
    data = json.load(fp)
    
record = data[0]
print("\nIn record:")
pprint(record)

out_record = process_record(tokenizer, record)

print("\nOut record:")
pprint(out_record)