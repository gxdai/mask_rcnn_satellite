#!/bin/bash
clear 
# old dataset
# TRAIN_JSON="/home/ubuntu/data/satellite/sate/annotations/instances_train2018.json"
# TEST_JSON="/home/ubuntu/data/satellite/sate/annotations/instances_test2018.json"
# IMAGE_ROOT_DIR="/home/ubuntu/data/satellite/sate"



# parameter settings
CHECKPOINT_DIR="logs/model_search"
DATASET_NAME="satellite"
TRAIN_JSON="/home/ubuntu/data/complete_set/annotations/complete.json"
TEST_JSON="/home/ubuntu/data/complete_set/annotations/val.json"
SKIP_CLASSES="Road Body_Of_water Tree"
IMAGE_ROOT_DIR="/home/ubuntu/data/complete_set/images"
INSTANCE_NUMBER_THRESHOLD=50
MODE="train"             # 1 for training, 0 for testing
DUP_FLAG=1







if [[ $(hostname) = "uranus" ]]; then
echo "Add conda path"
source $HOME/anaconda3/etc/profile.d/conda.sh

echo "Activate py36"
source $HOME/anaconda3/bin/activate py36

echo "export cuda path"
export LD_LIBRARY_PATH=/home/gxdai/cudnn6/lib64:/usr/local/cuda-9.0/lib64:$LD_LIBRARY_PATH

CUDA_VISIBLE_DEVICES=$1 py_gxdai $2 
elif [[ $(hostname) = "gxdai-Precision-7920-Tower" ]]; then

echo "Add conda path"
source $HOME/anaconda3/etc/profile.d/conda.sh

echo "Activate py36"
source $HOME/anaconda3/bin/activate py36


CUDA_VISIBLE_DEVICES=$1 python $2 
# CUDA_VISIBLE_DEVICES=$1 python $2 
elif [[ $(hostname) = "dgx-r103-2" ]]; then

echo "Add conda path"
source $HOME/anaconda3/etc/profile.d/conda.sh

echo "Activate py36"
source $HOME/anaconda3/bin/activate py36

CUDA_VISIBLE_DEVICES=$1 py_gxdai $2 --dataset_name $DATASET_NAME \
                                  --dup_flag $DUP_FLAG \
                                  --train_json $TRAIN_JSON \
                                  --test_json $TEST_JSON \
                                  --image_root_dir $IMAGE_ROOT_DIR \
                                  --checkpoint_dir $CHECKPOINT_DIR \
                                  --instance_number_threshold $INSTANCE_NUMBER_THRESHOLD \
                                  --mode $MODE



fi
