#!/bin/bash
module load libfabric/1.15.2.0
module list
##SBATCH -p a5
##SBATCH -x sls-sm-1,sls-2080-[1,3],sls-1080-[1,2,3],sls-sm-[5,6,7,12]
#SBATCH --gres=gpu:4
#SBATCH -c 4
#SBATCH -n 1
#SBATCH --mem=120000
#SBATCH --job-name="as-pretrain"
#SBATCH --output=../log/%j_as_pretrain.txt


# run cav-mae pretraining, use smaller lr and batch size, fits smaller GPUs (4*12GB GPUs)
export NCCL_COLLNET_ENABLE=1
export NCCL_NET_GDR_LEVEL=PHB
export TORCH_DISTRIBUTED_DEBUG=INFO
export NCCL_NSOCKS_PERTHREA = 4
export NCCL_SOCKET_NTHREADS = 2
NHOSTS=$(wc -l < "${PBS_NODEFILE}")
NGPU_PER_HOST=$(nvidia-smi -L | wc -l)
NGPUS="$((${NHOSTS}*${NGPU_PER_HOST}))"

set -x
# . /data/sls/scratch/share-201907/slstoolchainrc
# source /data/sls/scratch/yuangong/avbyol/venv-a5/bin/activate
# export TORCH_HOME=../../pretrained_models
cd /home/ben2002chou/code/cav-mae-pl/cav-mae
source cavmae1017/bin/activate
export TORCH_HOME=/home/ben2002chou/code/cav-mae-pl/cav-mae/pretrained_model
export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15'

model=cav-mae
masking_ratio=0.75
mask_mode=unstructured # or time, or freq, or tf
contrast_loss_weight=0.01 # cavmae uses 0.01
mae_loss_weight=1.0 # cavmae uses 1.0
mae_loss_weight=1.0 # cavmae uses 1.0
tr_pos=False
norm_pix_loss=True

cur_dir=$(pwd)
wget -nc https://www.dropbox.com/s/9nlz523a5q52w86/ori_mae_11.pth?dl=1 -O IN-initial.pth
pretrain_path=${cur_dir}/IN-initial.pth

bal=None
lr=5e-5
epoch=25
lrscheduler_start=10
lrscheduler_decay=0.5
lrscheduler_step=5
dataset_mean=-5.081
dataset_std=4.4849
target_length=1024
noise=True
mixup=0.0
batch_size=16 # use 48 to avoid memory error # no MAE can handle up to 128
lr_adapt=False

dataset=audioset
# tr_data=/home/ben2002chou/code/cav-mae/data/cocochorals/audioset_2m_cocochorals_train.json
tr_data=/home/ben2002chou/code/cav-mae/data/cocochorals/audioset_20k_cocochorals_test.json
te_data=/home/ben2002chou/code/cav-mae/data/cocochorals/audioset_eval_cocochorals_valid.json
label_csv=/home/ben2002chou/code/cav-mae/data/cocochorals/class_labels_indices_combined.csv

exp_dir=./exp_midi/testmae02-${dataset}-${model}-bal${bal}-lr${lr}-epoch${epoch}-bs${batch_size}-ngpus${NGPUS}-norm${norm_pix_loss}-c${contrast_loss_weight}-p${mae_loss_weight}-tp${tr_pos}-mr-${mask_mode}-${masking_ratio}
mkdir -p $exp_dir

mpiexec \
--verbose \
--envall \
--env CUDA_CACHE_DISABLE=1 \
-n "${NGPUS}" \
--ppn "${NGPU_PER_HOST}" \
--hostfile="${PBS_NODEFILE}" \
python -W ignore src/run_cavmae_pretrain_piano_roll.py --model ${model} --dataset ${dataset} \
--pretrain_path ${pretrain_path} \
--data-train ${tr_data} --data-val ${te_data} --exp-dir $exp_dir \
--label-csv ${label_csv} --n_class 527 \
--lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model True \
--mixup ${mixup} --bal ${bal} \
--lrscheduler_start ${lrscheduler_start} --lrscheduler_decay ${lrscheduler_decay} --lrscheduler_step ${lrscheduler_step} \
--dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --target_length ${target_length} --noise ${noise} --warmup True \
--lr_adapt ${lr_adapt} \
--norm_pix_loss ${norm_pix_loss} \
--mae_loss_weight ${mae_loss_weight} --contrast_loss_weight ${contrast_loss_weight} \
--tr_pos ${tr_pos} --masking_ratio ${masking_ratio} --mask_mode ${mask_mode}

# CUDA_CACHE_DISABLE=1 