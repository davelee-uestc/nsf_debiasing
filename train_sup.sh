#!/bin/bash
nvidia_smi_output=$(nvidia-smi|grep MiB |egrep -o '[0-9]+MiB /'|egrep -o [0-9]+)
idle_gpu_id=-1
idx=-1
idle_gpu_ids=""

while read -r line; do
    idx=$(($idx + 1))
    #if [[ $line == *" 0% "* ]] && [[ $line == *" 8MiB "* ]]; then
    if [ $line -lt 100 ] ;then
        idle_gpu_id=$idx
        idle_gpu_ids=("$idle_gpu_ids,$idx")
    fi
done <<< "$nvidia_smi_output"

if [ $idle_gpu_id -ge 0 ]; then
    echo "All idle GPUs: $idle_gpu_ids"
else
    echo "No idle GPUs found."
    exit 1
fi

#!/bin/bash

# Function to display usage
usage() {
    echo "Usage: $0 -d <dataset> [-b <batchsize>] [-n <n_epoch>] [-s <seed>]"
    echo "  -d <dataset>   : Required. Dataset name. Options are 'ce', 'wb', 'mn', 'cc', 'cx'."
    echo "  -b <batchsize> : Optional. Batch size. Default depends on dataset."
    echo "  -n <n_epoch>   : Optional. Number of epochs. Default depends on dataset."
    echo "  -s <seed>      : Optional. Seed for randomness. Default depends on dataset."
    exit 1
}

# Parse command line arguments
while getopts "d:b:n:s:" opt; do
    case ${opt} in
        d )
            DATASET=$OPTARG
            ;;
        b )
            BATCHSIZE=$OPTARG
            ;;
        n )
            N_EPOCH=$OPTARG
            ;;
        s )
            SEED=$OPTARG
            ;;
        * )
            usage
            ;;
    esac
done

# Check if required parameter is provided
if [ -z "$DATASET" ]; then
    echo "Error: Dataset is required."
    usage
fi

# Validate dataset value and set default parameters based on dataset
case $DATASET in
    "mm")
        DEFAULT_BATCHSIZE=100
        DEFAULT_N_EPOCH=20
        DEFAULT_SEED=1
        ;;
    "cx")
        DEFAULT_BATCHSIZE=100
        DEFAULT_N_EPOCH=20
        DEFAULT_SEED=1
        ;;
    "ce")
        DEFAULT_BATCHSIZE=100
        DEFAULT_N_EPOCH=20
        DEFAULT_SEED=1
        ;;
    "wb")
        DEFAULT_BATCHSIZE=32
        DEFAULT_N_EPOCH=100
        DEFAULT_SEED=1
        ;;
    "mn")
        DEFAULT_BATCHSIZE=16
        DEFAULT_N_EPOCH=10
        DEFAULT_SEED=1
        ;;
    "cc")
        DEFAULT_BATCHSIZE=16
        DEFAULT_N_EPOCH=10
        DEFAULT_SEED=1
        ;;
    *)
        echo "Error: Invalid dataset. Options are 'ce', 'wb', 'mn', 'cc', 'cx','mm'."
        usage
        ;;
esac

# Assign default values if not provided by the user
BATCHSIZE=${BATCHSIZE:-$DEFAULT_BATCHSIZE}
N_EPOCH=${N_EPOCH:-$DEFAULT_N_EPOCH}
SEED=${SEED:-$DEFAULT_SEED}

# Print parameters
echo "Running script with the following parameters:"
echo "Dataset: $DATASET"
echo "Batch size: $BATCHSIZE"
echo "Number of epochs: $N_EPOCH"
echo "Seed: $SEED"
echo "GPU: $idle_gpu_id"

# Your script logic goes here
# For example, you can use these parameters in your training script
# python train.py --dataset $DATASET --batchsize $BATCHSIZE --n_epoch $N_EPOCH --seed $SEED
case $DATASET in
    "cx")
CUDA_VISIBLE_DEVICES=$idle_gpu_id nohup python3 train_supervised.py --output_dir=logs/chexpert/erm_seed$SEED \
	--num_epochs=$N_EPOCH --eval_freq=1 --save_freq=100 --seed=$SEED \
	--weight_decay=1e-4 --batch_size=$BATCHSIZE --init_lr=3e-3 \
	--scheduler=cosine_lr_scheduler --data_dir=chexpert \
	--data_transform=AugWaterbirdsCelebATransform \
	--dataset=SpuriousCorrelationDataset --model=imagenet_resnet50_pretrained --label_filename metadata.csv > cx_sup_train_$idle_gpu_id.log 2>&1 &
        ;;
    "mm")
CUDA_VISIBLE_DEVICES=$idle_gpu_id nohup python3 train_supervised.py --output_dir=logs/mimic/erm_seed$SEED \
	--num_epochs=$N_EPOCH --eval_freq=1 --save_freq=100 --seed=$SEED \
	--weight_decay=1e-4 --batch_size=$BATCHSIZE --init_lr=3e-3 \
	--scheduler=cosine_lr_scheduler --data_dir=MIMIC-CXR-JPG \
	--data_transform=AugWaterbirdsCelebATransform \
	--dataset=SpuriousCorrelationDataset --model=imagenet_resnet50_pretrained --label_filename metadata.csv > mm_sup_train_$idle_gpu_id.log 2>&1 &
        ;;
    "ce")
CUDA_VISIBLE_DEVICES=$idle_gpu_id nohup python3 train_supervised.py --output_dir=logs/celeba/erm_seed$SEED \
	--num_epochs=$N_EPOCH --eval_freq=1 --save_freq=100 --seed=$SEED \
	--weight_decay=1e-4 --batch_size=$BATCHSIZE --init_lr=3e-3 \
	--scheduler=cosine_lr_scheduler --data_dir=celeba \
	--data_transform=AugWaterbirdsCelebATransform \
	--dataset=SpuriousCorrelationDataset --model=imagenet_resnet50_pretrained --label_filename metadata.csv > ce_sup_train_$idle_gpu_id.log 2>&1 &
        ;;
    "wb")
        CUDA_VISIBLE_DEVICES=$idle_gpu_id nohup python3 train_supervised.py --output_dir=logs/waterbirds/erm_seed$SEED \
	--num_epochs=$N_EPOCH --eval_freq=1 --save_freq=100 --seed=$SEED \
	--weight_decay=1e-4 --batch_size=$BATCHSIZE --init_lr=3e-3 \
	--scheduler=cosine_lr_scheduler --data_dir=waterbirds \
	--data_transform=AugWaterbirdsCelebATransform \
	--dataset=SpuriousCorrelationDataset --model=imagenet_resnet50_pretrained --label_filename  metadata.csv > wb_sup_train_$idle_gpu_id.log 2>&1 &
        ;;
    "mn")
 CUDA_VISIBLE_DEVICES=$idle_gpu_id nohup  python3 train_supervised.py --output_dir=logs/multinli/erm_seed$SEED/ \
	--num_epochs=$N_EPOCH --eval_freq=1 --save_freq=10 --seed=$SEED \
	--weight_decay=1.e-4 --batch_size=$BATCHSIZE --init_lr=1e-5 \
	--scheduler=bert_lr_scheduler --data_dir=multinli \
	--data_transform=None --dataset=MultiNLIDataset --model=bert_pretrained \
	--optimizer=bert_adamw_optimizer > mn_sup_train_$idle_gpu_id.log 2>&1 &
        ;;
    "cc")
	CUDA_VISIBLE_DEVICES=$idle_gpu_id nohup  python3 train_supervised.py --output_dir=logs/civilcomments/erm_seed$SEED \
	--num_epochs=$N_EPOCH --eval_freq=1 --save_freq=10 --seed=$SEED \
	--weight_decay=1.e-4 --batch_size=$BATCHSIZE --init_lr=1e-5 \
	--scheduler=bert_lr_scheduler --data_dir=cc \
	--data_transform=BertTokenizeTransform \
	--dataset=WildsCivilCommentsCoarse --model=bert_pretrained \
	--optimizer=bert_adamw_optimizer  > cc_sup_train_$idle_gpu_id.log 2>&1 &
        ;;
    *)
        echo "Error: Invalid dataset. Options are 'ce', 'wb', 'mn', 'cc', 'cx'."
        usage
        ;;
esac