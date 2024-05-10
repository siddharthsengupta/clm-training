WORKDIR_PATH=/opt/ml

set -e

unzip $WORKDIR_PATH/input/data/model/dpr_$1 -d $WORKDIR_PATH/input/data/model/

python3 $WORKDIR_PATH/dpr_training/train.py \
        --output_dir $WORKDIR_PATH/model/dpr_$1 \
        --query_model_name_or_path $WORKDIR_PATH/input/data/model/dpr_$1/query_encoder \
        --passage_model_name_or_path $WORKDIR_PATH/input/data/model/dpr_$1/passage_encoder \
        --data_dir $WORKDIR_PATH/input/data/train \
        --train_file dpr_$1_train.json \
        --predict_file dpr_$1_test.json \

cd $WORKDIR_PATH/model
pwd
ls -la
zip -r dpr_$1.zip dpr_$1/*


# bash run.sh small
# bash run.sh long
