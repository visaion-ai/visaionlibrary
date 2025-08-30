GPU_IDS=$1
PYTHON_INTERPRETER=$2
SCRIPT_PATH=$3
CONFIG_PATH=$4
GPUS=$5
PORT=$6
# echo "CUDA_VISIBLE_DEVICES=$GPU_IDS $PYTHON_INTERPRETER -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT $SCRIPT_PATH $CONFIG_PATH --launcher pytorch"
CUDA_VISIBLE_DEVICES=$GPU_IDS $PYTHON_INTERPRETER -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT $SCRIPT_PATH $CONFIG_PATH --launcher pytorch 
