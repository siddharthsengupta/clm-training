WORKDIR_PATH=/opt/ml

set -e

# export NCCL_SHM_DISABLE=1

unzip $WORKDIR_PATH/input/data/model/dpr_$1 -d $WORKDIR_PATH/input/data/model/

python3 $WORKDIR_PATH/dpr_training/train.py \
        --output_dir $WORKDIR_PATH/model/dpr_$1 \
        --use_base_model $2 \
        --query_model_name_or_path $WORKDIR_PATH/input/data/model/dpr_$1/query_encoder \
        --passage_model_name_or_path $WORKDIR_PATH/input/data/model/dpr_$1/passage_encoder \
        --data_dir $WORKDIR_PATH/input/data \
        --train_file train/dpr_$1_train.json \
        --predict_file test/dpr_$1_test.json \

cd $WORKDIR_PATH/model
zip -r dpr_$1.zip dpr_$1/*
rm -rf dpr_$1/


# bash run.sh small False
# bash run.sh long False
