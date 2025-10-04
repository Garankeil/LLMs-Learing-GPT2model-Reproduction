# LLMs-Learing-GPT2model-Reproduction
My frist learing repository, save my learning process, learning code and some results.

- GPT2 model created from bilibili.chaofa用代码打点酱油
- Use DistributedDataParallel(DDP) to train my GPT2 on two Nvidia RTX 2080Ti(22GB)
- Use memory map to load the seqmonkey dataset(32GB)(1 json chunk to 25 npys)
- Use argparse to manage parameters
- Add logs to tensorboard for visualizing the training process

## Model coding
- Traditional MHA:
  - Single Head Attention->Muti Head Attention
  
- Params from GPT2-Small (dims=768, max_seq_len=512, N_blocks = 12 ...)
  > in model/Transformers.py

- Use kv cache to speed inference process
  - just in GPT.generate(x), x's length was cut to 1
  - the k and v matrix of the past token was cached
  - when new token was put in attention process, the new k and v matrix of new token concat with kv_old
 
- Use GroupQueryAttention(GQA) instead of MHA
  - reduce the K and V heads to half or less of Q heads
  - When calculating the attention score, the head number of K and V was repeated to heads of Q  


## Dataset prepocessing
- seqmonkey chinese dataset has 13000000 lines and more than 32GB, so I choose to chunk the json file to 25 npy files.
  > in dataset_chunk.py
  - Cut 2e6 lines from seqmonkey for pretrain
  - Each npy file has 2e5 lines
- Each tokens has a length of 513 (shift operation)
  - token[:-1] for model input
  - token[1:] for the target

## Dataset loader
> in train.py(class LineChunkDataset)
- Use memory-map technology from numpy to reduce RAM usage (compares to load all data into RAM)
  - Load data from disk on demand only, with a constant memory usage
  - Compared to streaming loading, the memory-map method supports random access and can predict the data length
 
## Distributed Data Parallel (DDP)
> in train.py
- in the past, I use dataparallel to train my model on two GPU, but it only put half of dataset to another GPU, while the backward propagation is still completed on one GPU
- DDP is a more efficient method：
  - Each GPU has the same model and params on the beginning.
  - After the training begins, two GPUs each calculate the forward propagation, and during backpropagation, they synchronize gradients through nccl communication to achieve consistent parameters. 
  - Gradient transfer is performed simultaneously with backward propagation. (All reduce)
  - Each GPU has the gradient from other GPUs, and calculate the avarage of the gradients to update the model params.
  > Offcial document of Pytorch (https://docs.pytorch.org/docs/stable/notes/ddp.html)


# Usage:
> torchrun --nproc_per_node=2 train.py


![s3794979](https://github.com/user-attachments/assets/fd161ec4-a700-4c40-be3c-33478ae8b037)
