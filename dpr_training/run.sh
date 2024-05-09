PATH=/opt/ml

unzip $PATH/input/data/model/dpr_$1 -d $PATH/input/data/model/

python3 $PATH/dpr_training/train.py \
        --output_dir $PATH/model/dpr_$1 \
        --query_model_name_or_path $PATH/input/data/model/dpr_$1/query_encoder \
        --passage_model_name_or_path $PATH/input/data/model/dpr_$1/passage_encoder \
        --data_dir $PATH/input/data/train \
        --train_file dpr_$1_train.json \
        --predict_file dpr_$1_test.json \

cd $PATH/model/dpr_$1
pwd
zip -r dpr_$1.zip *


# bash run.sh small
# bash run.sh long