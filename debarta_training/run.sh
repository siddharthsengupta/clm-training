WORKDIR_PATH=/opt/ml

set -e

unzip $WORKDIR_PATH/input/data/model/debarta_$1 -d $WORKDIR_PATH/input/data/model/

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 $WORKDIR_PATH/debarta_training/train.py \
        --output_dir $WORKDIR_PATH/model/debarta_$1 \
        --model_type debarta \
        --use_base_model $2 \
        --model_name_or_path $WORKDIR_PATH/input/data/model/debarta_$1 \
        --train_file $WORKDIR_PATH/input/data/train/debarta_$1_train.json \
        --predict_file $WORKDIR_PATH/input/data/test/debarta_$1_test.json \
        --do_train \
        --do_eval \
        --version_2_with_negative \
        --max_seq_length $3 \
        --max_answer_length $3 \
        --doc_stride $4 \
        --overwrite_output_dir \
        # --learning_rate 1e-4 \
        # --num_train_epochs 4 \
        # --per_gpu_eval_batch_size=2  \
        # --per_gpu_train_batch_size=2 \
        # --save_steps 1000 \
        # --n_best_size 20 \
        # --gradient_accumulation_steps 4\
        

cd $WORKDIR_PATH/model
zip -r debarta_$1.zip debarta_$1/*
rm -rf debarta_$1/


# bash run.sh small false 512 256
# bash run.sh long_1 false 1024 512
# bash run.sh long_2 false 1024 512
