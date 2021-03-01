# ConformerSED
Implementation of the paper [CONFORMER-BASED SOUND EVENT DETECTION WITH
SEMI-SUPERVISED LEARNING AND DATA AUGMENTATION](http://dcase.community/documents/workshop2020/proceedings/DCASE2020Workshop_Miyazaki_92.pdf)

## Dataset
To run this repository's script, please make sure the dataset is cofigured as follows:
```
└── dcase20_task4
     └── dataset
         └── audio
             ├── train
             |   ├── synthetic20
             |   ├── weak
             |   └── unlabel_in_domain
             ├── validation
             └── public
```
To acquire the dataset, run stage 1 in `bin/run.sh` or download directly.

## Quick start
```bash
$ conda env create -f environment.yaml
$ conda activate dcase20t4
$ wandb login <YOUR_API_KEY>
$ bash ./bin/run.sh
```

## Results
|     |Event-based macro F1|PSDS|
|:---:|---:|---:|
|Validation|47.7%|0.637|
|Public|49.0%|0.681|

To reproduce the result on public eval set, download pretrained model and config file [here](https://drive.google.com/file/d/1dBfwtKSCLxG6dkYDlai3a3nD-Z6D_4yB/view?usp=sharing).

Then, put model and config as follows:
```
├── data
|    └── feat
|        └── sr16000_n_mels64_n_fft1024_n_shift323
|            └── public
└── exp
    └── conformer_sed                                  
        ├── model
        │   └── model_best_score.pth
        ├── model_config.yaml
        ├── trainer_config.yaml
        ├── post_process_params.pickle
        └── stats.npz
```
After that, you can run test script
```bash
$ python ./src/test.py
```

## Disclaimer
- Some audio samples in the original dataset may be missing for some reason (e.g., removed on the internet). For audio samples used in this experiment, see metadata.
- This repository results are slightly better than reported in the paper.
## Contact
If you have any questions, please feel free to ask me.

Koichi Miyazaki (E-mail: miyazaki.koichi_at_g.sp.m.is.nagoya-u.ac.jp)

## Reference
DCASE2020 Task4 baseline: https://github.com/turpaultn/dcase20_task4
