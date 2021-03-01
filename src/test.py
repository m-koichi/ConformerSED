#!/usr/bin/env python
# encoding: utf-8

# Copyright 2020 Koichi Miyazaki (Nagoya University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import logging
import math
import os
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from baseline_utils.ManyHotEncoder import ManyHotEncoder
from dataset import SEDDataset
from models.sed_model import SEDModel
from post_processing import PostProcess
from trainer import MeanTeacherTrainerOptions
from transforms import get_transforms


def seed_everything(seed):
    logging.info("random seed = %d" % seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_name", default="conformer_sed", type=str, help="exp name used for the training")
    parser.add_argument("--debugmode", default=True, action="store_true", help="Debugmode")
    parser.add_argument("--verbose", "-V", default=0, type=int, help="Verbose option")
    parser.add_argument("--test_meta", default="./data/metadata/public/public.tsv")
    parser.add_argument("--test_audio_dir", default="./dcase20_task4/dataset/audio/public")

    return parser.parse_args(args)


def test(model, test_loader, output_dir, options, pp_params={}):
    post_process = PostProcess(model, test_loader, output_dir, options)
    post_process.show_best(pp_params)
    post_process.compute_psds()


def main(args):
    args = parse_args(args)

    exp_name = Path(f"./exp/{args.exp_name}")
    assert exp_name.exists()

    # load config
    with open(exp_name / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    test_df = pd.read_csv(args.test_meta, header=0, sep="\t")

    n_frames = math.ceil(
        cfg["max_len_seconds"] * cfg["feature"]["sample_rate"] / cfg["feature"]["mel_spec"]["hop_size"]
    )
    # Note: assume that the same class used in the training is included at least once.
    classes = test_df.event_label.dropna().sort_values().unique()
    many_hot_encoder = ManyHotEncoder(labels=classes, n_frames=n_frames)
    encode_function = many_hot_encoder.encode_strong_df

    feat_dir = Path(
        f"data/feat/sr{cfg['feature']['sample_rate']}_n_mels{cfg['feature']['mel_spec']['n_mels']}_"
        + f"n_fft{cfg['feature']['mel_spec']['n_fft']}_hop_size{cfg['feature']['mel_spec']['hop_size']}"
    )
    stats = np.load(
        f"exp/{cfg['exp_name']}/stats.npz",
    )

    norm_dict_params = {
        "mean": stats["mean"],
        "std": stats["std"],
        "mode": cfg["norm_mode"],
    }

    test_transforms = get_transforms(
        cfg["data_aug"],
        nb_frames=n_frames,
        norm_dict_params=norm_dict_params,
        training=False,
        prob=0.0,
    )

    test_dataset = SEDDataset(
        test_df,
        data_dir=feat_dir / "public",
        encode_function=encode_function,
        pooling_time_ratio=cfg["pooling_time_ratio"],
        transforms=test_transforms,
    )

    if cfg["ngpu"] > 1:
        cfg["batch_size"] *= cfg["ngpu"]

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )

    # logging info
    if args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARN,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        logging.warning("Skip DEBUG/INFO messages")

    # display PYTHONPATH
    logging.info("python path = " + os.environ.get("PYTHONPATH", "(None)"))

    seed_everything(cfg["seed"])

    model = SEDModel(n_class=10, cnn_kwargs=cfg["model"]["cnn"], encoder_kwargs=cfg["model"]["encoder"])

    checkpoint = torch.load(exp_name / "model" / "model_best_score.pth")
    model.load_state_dict(checkpoint["state_dict"])

    trainer_options = MeanTeacherTrainerOptions(**cfg["trainer_options"])
    trainer_options._set_validation_options(
        valid_meta=args.test_meta,
        valid_audio_dir=args.test_audio_dir,
        max_len_seconds=cfg["max_len_seconds"],
        sample_rate=cfg["feature"]["sample_rate"],
        hop_size=cfg["feature"]["mel_spec"]["hop_size"],
        pooling_time_ratio=cfg["pooling_time_ratio"],
    )
    output_dir = exp_name / "test"
    output_dir.mkdir(exist_ok=True)
    with open(exp_name / "post_process_params.pickle", "rb") as f:
        pp_params = pickle.load(f)

    model = model.to(trainer_options.device)
    test(model, test_loader, output_dir, trainer_options, pp_params=pp_params)


if __name__ == "__main__":
    main(sys.argv[1:])
