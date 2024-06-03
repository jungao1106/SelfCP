export CUDA_VISIBLE_DEVICES=4,5,6,7
#NCCL_BLOCKING_WAIT=1 NCCL_ASYNC_ERROR_HANDLING=1 OMP_NUM_THREADS=50 nohup torchrun --nnodes=1 --nproc_per_node=4 --master_port=11006 /data1/gj/PromptCompression/main.py >./ratio-select.txt 2>&1 & #脱机
NCCL_BLOCKING_WAIT=1 NCCL_ASYNC_ERROR_HANDLING=1 OMP_NUM_THREADS=50 torchrun --nnodes=1 --nproc_per_node=4 --master_port=11006 /data1/gj/PromptCompression/main.py
