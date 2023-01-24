while getopts m:c:g: flag
do
    case "${flag}" in
        m) model=${OPTARG};;
        c) ncpu=${OPTARG};;
        g) ngpu=${OPTARG};;
    esac
done

folder="configs/$model"
for entry in "$folder"/*
do
    echo "$entry"
    CUDA_VISIBLE_DEVICES=4,5,6,7 python tune.py --config $entry  --n_cpu $ncpu --n_gpu $ngpu --n_trials 30
done