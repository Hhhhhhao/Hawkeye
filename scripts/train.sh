model='baseline_r50_in1k_224'
n_gpu=1

declare -i cnt
cnt=0
folder="configs/$model"
for entry in "$folder"/*
do
    echo "$entry"
    CUDA_VISIBLE_DEVICES=$cnt nohup python train.py --config $entry &
    cnt+=$n_gpu
done