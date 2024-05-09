PATH=/opt/ml

unzip $PATH/input/data/model/debarta_$1 -d $PATH/input/data/model/

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 $PATH/debarta_training/train.py \
        --output_dir $PATH/model/debarta_$1 \
        --model_type debarta \
        --model_name_or_path $PATH/input/data/model/debarta_$1 \
        --train_file $PATH/input/data/train/debarta_$1_train.json \
        --predict_file $PATH/input/data/test/debarta_$1_test.json \
        --do_train \
        --do_eval \
        --version_2_with_negative \
        --learning_rate 1e-4 \
        --num_train_epochs 4 \
        --per_gpu_eval_batch_size=2  \
        --per_gpu_train_batch_size=2 \
        --max_seq_length $2 \
        --max_answer_length $2 \
        --doc_stride $3 \
        --save_steps 1000 \
        --n_best_size 20 \
        --gradient_accumulation_steps 4\
        --overwrite_output_dir

cd $PATH/model/debarta_$1
pwd
zip -r debarta_$1.zip *


# bash run.sh small 512 256
# bash run.sh long_1 1024 512
# bash run.sh long_2 1024 512