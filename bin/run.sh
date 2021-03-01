#!/bin/bash

# Copyright 2020 Nagoya University (Koichi Miyazaki)
# MIT  (https://opensource.org/licenses/MIT)

stage=1         # start from 0 if you need to start from data preparation
stop_stage=4
nj=24
config=./config/default_config.yaml
 

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: Data Preparation"
    if [ ! -d dcase20_task4 ]; then
        git clone git@github.com:turpaultn/dcase20_task4.git
        cd dcase20_task4/scripts
        ./1_download_data.sh
        ./2_generate_data_from_jams.sh
        # [optional] generate augmented samples
        # ./3_reverberate_data.sh
        # ./4_separate_mixture.sh
        cd ..
        cd dcase20_task4
        wget https://zenodo.org/record/3588172/files/DESEDpublic_eval.tar.gz -O ./DESEDpublic_eval.tar.gz
        tar -xzvf DESEDpublic_eval.tar.gz
        cd ../
    fi
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: Feature Generation"
    python local/feature_extraction.py --config ${config} --nj ${nj}
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    echo "stage 3: Network Training"
    echo "`date`: [Start] model training"
    python -u src/train.py --config ${config}
    echo "`date`: [Done] model training"

fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Evaluate performance"
    echo "`date`: [Start] model evaluation"
    python -u src/test.py
    echo "`date`: [Done] model evaluation"
fi
