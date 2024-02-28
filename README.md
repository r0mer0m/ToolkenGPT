# ToolkenGPT
**Source code for [ToolkenGPT: Augmenting Frozen Language Models with Massive Tools via Tool Embeddings](https://arxiv.org/abs/2305.11554)**

[NeurIPS 2023 (oral)](https://nips.cc/Conferences/2023) | [Best Paper Award at SoCalNLP 2023](https://socalnlp.github.io/symp23/index.html)

![Figure](assets/image.png)

## Preparation
+ Our experiments are conducted with LLaMA-13B/33B, which takes at least 2/4 GPUs of 24GB memory each.
+ Acquire the checkpoints of LLaMA from MetaAI and install all required packages. Please refer to [LLaMA official repo](https://github.com/facebookresearch/llama).
+ Download the data from [here](https://drive.google.com/file/d/13Sj7uIsyqWXoTh1ejWUviTzeQSES2Omd/view?usp=sharing) (all datasets uploaded)
+ (For VirtualHome) Please download the data following the instructions [here](virtualhome/README.md).
    > A side note: the folder `virtualhome` is from its [official repo](https://github.com/xavierpuigf/virtualhome), but we fixed some small bugs in the evolving graph.

## GSM8K-XL

### Train

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node 4 --master_port 1200 train_llama.py --ckpt_dir $LLAMA_CKPTS/30B --tokenizer_path $LLAMA_CKPTS/tokenizer.model --input_file data/gsm8k-xl/train.json --lr 1e-3 --num_epochs 10
```

### Inference

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node 4 --master_port 1250 inference_llama.py --ckpt_dir $LLAMA_CKPTS/30B --tokenizer_path $LLAMA_CKPTS/tokenizer.model --mode func_embedding --dataset gsm8k-xl  --func_load_path checkpoints/gsm8k-xl/epoch_3.pth --logits_bias 3.0
```

## FuncQA

### Train

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node 4 --master_port 1200 train_llama.py --ckpt_dir $PATH_TO_LLAMA/30B --tokenizer_path $PATH_TO_LLAMA/tokenizer.model --input_file data/funcqa/train.json --lr 1e-4 --num_epochs 10
```

### Inference (1-hop)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node 4 --master_port 1250 inference_llama.py --ckpt_dir $LLAMA_CKPTS/30B --tokenizer_path $LLAMA_CKPTS/tokenizer.model --mode func_embedding --dataset funcqa_oh --func_load_path checkpoints/funcqa/epoch_7.pth --logits_bias 2.7
```

### Inference (MultiHop)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node 4 --master_port 1250 inference_llama.py --ckpt_dir $LLAMA_CKPTS/30B --tokenizer_path $LLAMA_CKPTS/tokenizer.model --mode func_embedding --dataset funcqa_mh --func_load_path checkpoints/funcqa/epoch_7.pth --logits_bias 4.0
```

## VirtualHome

### Training
```bash
python -m torch.distributed.run --nproc_per_node 2 --master_port 3001 train_llama.py --ckpt_dir $LLAMA_CKPTS/13B --tokenizer_path $LLAMA_CKPTS/tokenizer.model --dataset vh --input_file data/vh/legal_train_v4_embedding.json --only_functoken True --num_epochs 10
```


### Inference

```bash
CUDA_VISIBLE_DEVICES=3,5 python -m torch.distributed.run --nproc_per_node 2 inference_llama.py --ckpt_dir $LLAMA_CKPTS/13B --tokenizer_path $LLAMA_CKPTS/tokenizer.model --mode vh_embedding_inference --dataset vh --func_load_path checkpoints/vh/epoch_7.pth --logits_bias 10.0
```

### Evaluation

See `evaluation/eval_vh.ipynb`

## KAMEL
### Train
+ synthetic data
```bash
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.run --nproc_per_node 2 --master_port 3002 train_llama.py --ckpt_dir $LLAMA_CKPTS/13B --tokenizer_path $LLAMA_CKPTS/tokenizer.model --dataset kamel --input_file data/kamel/train_clean.json --only_functoken False ---log_every 500 --num_epochs 10
```


+ supervised data
```bash
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.run --nproc_per_node 2 --master_port 3002 train_llama.py --ckpt_dir $LLAMA_CKPTS/13B --tokenizer_path $LLAMA_CKPTS/tokenizer.model --dataset kamel --input_file data/kamel/kamel_id_train.json --only_functoken False ---log_every 500 --num_epochs 10
```

### Inference

```bash
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.run --nproc_per_node 2 inference_llama.py --ckpt_dir $LLAMA_CKPTS/13B --tokenizer_path $LLAMA_CKPTS/tokenizer.model --mode kamel_embedding_inference --dataset kamel_30 --func_load_path checkpoints/kamel/epoch_4.pth --logits_bias 10
```

### Evaluation

See `evaluation/eval_kamel.ipynb`




---------------------------------------------


```
<!-- conda install cuda -c nvidia/label/cuda-11.6.0 -->
conda install cuda -c nvidia
conda install conda-forge::deepspeed
conda install lightning -c conda-forge

conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia # does not install cuda version
```
conda install -c conda-forge gxx
conda install conda-forge::ninja
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116


?? conda install nvidia/label/cuda-11.4.0::cuda-nvcc


```bash
export LLAMA_CKPTS=../models/llama_checkpoints && echo $LLAMA_CKPTS
```


## Tokenizer Augmentation

```bash
cd ToolkenGPT
python scripts/augment_tokenizer.py --in_tok_path $LLAMA_CKPTS/tokenizer.model --new_tokens \<add\> \<subtract\> \<multiply\> \<divide\> \<power\> \<sqrt\> \<log\> \<ln\> \<lcm\> \<gcd\> \<remainder\> \<choose\> \<permutate\> --out_tok_dir ./augmented_tokenizer/
```

## Data Processing
```bash
python generate_data_funcqa.py
```


## GSM8K-XL

### Train

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node 2 --master_port 1200 train_augmented_llama.py --ckpt_dir $LLAMA_CKPTS/llama-2-13b-chat --tokenizer_path ./augmented_tokenizer --input_file ../augmented_data/gsm8k-xl/train.json --lr 1e-3 --num_epochs 10
```


## FuncQA

### Train

For debugging use `CUDA_LAUNCH_BLOCKING=1`.

2 nodes:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node 2 --master_port 1200 train_augmented_llama.py --ckpt_dir $LLAMA_CKPTS/llama-2-13b-chat --tokenizer_path ./augmented_tokenizer --input_file ../augmented_data/funcqa/train.json --lr 1e-4 --num_epochs 10 --dataset funcqa
```

4 nodes:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node 4 --master_port 1200 train_augmented_llama.py --ckpt_dir $LLAMA_CKPTS/llama-2-13b-chat --tokenizer_path ./augmented_tokenizer --input_file ../augmented_data/funcqa/train.json --lr 1e-4 --num_epochs 10 --dataset funcqa
```


```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m torch.distributed.run --nproc_per_node 8 --master_port 1200 train_augmented_llama.py --ckpt_dir $LLAMA_CKPTS/llama-2-13b-chat --tokenizer_path ./augmented_tokenizer --input_file ../augmented_data/funcqa/train.json --lr 1e-4 --num_epochs 10 --dataset funcqa
```