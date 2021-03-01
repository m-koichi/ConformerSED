#!/usr/bin/env python
# encoding: utf-8

# Copyright 2019 Koichi Miyazaki (Nagoya University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import logging
import math
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wandb
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from baseline_utils.ManyHotEncoder import ManyHotEncoder
from dataset import SEDDataset
from models.sed_model import SEDModel
from trainer import MeanTeacherTrainer, MeanTeacherTrainerOptions
from transforms import ApplyLog, Compose, get_transforms


def collect_stats(datasets, save_path):
    """Compute dataset statistics
    Args:
        dataset:
        save_path:
    Return:
        mean: (np.ndarray)
        std: (np.ndarray)
    """
    logging.info("compute dataset statistics")
    stats = {}
    for dataset in datasets:
        dataloader = DataLoader(dataset, batch_size=1)

        for x, _, _ in tqdm(dataloader):
            if len(stats) == 0:
                stats["mean"] = np.zeros(x.size(-1))
                stats["std"] = np.zeros(x.size(-1))
            stats["mean"] += x.numpy()[0, 0, :, :].mean(axis=0)
            stats["std"] += x.numpy()[0, 0, :, :].std(axis=0)
        stats["mean"] /= len(dataset)
        stats["std"] /= len(dataset)

    np.savez(save_path, **stats)

    return stats


def save_args(args, dest_dir, file_name="config.yaml"):
    import yaml

    print(yaml.dump(vars(args)))
    with open(os.path.join(dest_dir, file_name), "w") as f:
        f.write(yaml.dump(vars(args)))


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
    parser.add_argument("--config", default="./config/default_config.yaml", type=str, help="config file in yaml format")
    parser.add_argument("--debugmode", default=True, action="store_true", help="Debugmode")
    parser.add_argument("--verbose", "-V", default=0, type=int, help="Verbose option")

    return parser.parse_args(args)


def main(args):
    args = parse_args(args)

    # load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    wandb.init(project=cfg["wandb_project"], config=cfg, name=cfg["exp_name"])

    exp_name = Path(f"exp/{cfg['exp_name']}")
    # if debug is true, enable to overwrite experiment
    if exp_name.exists():
        logging.warning(f"{exp_name} is already exist.")
        if args.debugmode:
            logging.warning("Note that experiment will be overwrite.")
        else:
            logging.info("Experiment is interrupted. Make sure exp_name will be unique.")
            sys.exit(0)
    exp_name.mkdir(exist_ok=True)
    Path(exp_name / "model").mkdir(exist_ok=True)
    Path(exp_name / "predictions").mkdir(exist_ok=True)
    Path(exp_name / "log").mkdir(exist_ok=True)
    Path(exp_name / "score").mkdir(exist_ok=True)

    # save config
    shutil.copy(args.config, (exp_name / "config.yaml"))

    train_synth_df = pd.read_csv(cfg["synth_meta"], header=0, sep="\t")
    train_weak_df = pd.read_csv(cfg["weak_meta"], header=0, sep="\t")
    train_unlabel_df = pd.read_csv(cfg["unlabel_meta"], header=0, sep="\t")
    valid_df = pd.read_csv(cfg["valid_meta"], header=0, sep="\t")

    n_frames = math.ceil(
        cfg["max_len_seconds"] * cfg["feature"]["sample_rate"] / cfg["feature"]["mel_spec"]["hop_size"]
    )
    classes = valid_df.event_label.dropna().sort_values().unique()
    many_hot_encoder = ManyHotEncoder(labels=classes, n_frames=n_frames)
    encode_function = many_hot_encoder.encode_strong_df
    # Put train_synth in frames so many_hot_encoder can work.
    #  Not doing it for valid, because not using labels (when prediction) and event based metric expect sec.
    train_synth_df.onset = (
        train_synth_df.onset * cfg["feature"]["sample_rate"] // cfg["feature"]["mel_spec"]["hop_size"]
    )
    train_synth_df.offset = (
        train_synth_df.offset * cfg["feature"]["sample_rate"] // cfg["feature"]["mel_spec"]["hop_size"]
    )

    # For calculate validation loss. Note that do not use for calculate evaluation metrics
    valid_df.onset = valid_df.onset * cfg["feature"]["sample_rate"] // cfg["feature"]["mel_spec"]["hop_size"]
    valid_df.offset = valid_df.offset * cfg["feature"]["sample_rate"] // cfg["feature"]["mel_spec"]["hop_size"]

    feat_dir = Path(
        f"data/feat/sr{cfg['feature']['sample_rate']}_n_mels{cfg['feature']['mel_spec']['n_mels']}_"
        + f"n_fft{cfg['feature']['mel_spec']['n_fft']}_hop_size{cfg['feature']['mel_spec']['hop_size']}"
    )

    # collect dataset stats
    if Path(f"exp/{cfg['exp_name']}/stats.npz").exists():
        stats = np.load(
            f"exp/{cfg['exp_name']}/stats.npz",
        )
    else:
        kwargs_dataset = {
            "encode_function": encode_function,
            "transforms": Compose([ApplyLog()]),
        }
        train_synth_dataset = SEDDataset(train_synth_df, data_dir=(feat_dir / "train/synthetic20"), **kwargs_dataset)
        train_weak_dataset = SEDDataset(train_weak_df, data_dir=(feat_dir / "train/weak"), **kwargs_dataset)
        train_unlabel_dataset = SEDDataset(
            train_unlabel_df, data_dir=(feat_dir / "train/unlabel_in_domain"), **kwargs_dataset
        )
        stats = collect_stats(
            [train_synth_dataset, train_weak_dataset, train_unlabel_dataset],
            f"exp/{cfg['exp_name']}/stats.npz",
        )

    norm_dict_params = {
        "mean": stats["mean"],
        "std": stats["std"],
        "mode": cfg["norm_mode"],
    }

    train_transforms = get_transforms(
        cfg["data_aug"],
        nb_frames=n_frames,
        norm_dict_params=norm_dict_params,
        training=True,
        prob=cfg["apply_prob"],
    )
    test_transforms = get_transforms(
        cfg["data_aug"],
        nb_frames=n_frames,
        norm_dict_params=norm_dict_params,
        training=False,
        prob=0.0,
    )

    kwargs_dataset = {
        "encode_function": encode_function,
        "twice_data": True,
    }

    train_synth_dataset = SEDDataset(
        train_synth_df,
        data_dir=(feat_dir / "train/synthetic20"),
        encode_function=encode_function,
        pooling_time_ratio=cfg["pooling_time_ratio"],
        transforms=train_transforms,
        twice_data=True,
    )
    train_weak_dataset = SEDDataset(
        train_weak_df,
        data_dir=(feat_dir / "train/weak"),
        encode_function=encode_function,
        pooling_time_ratio=cfg["pooling_time_ratio"],
        transforms=train_transforms,
        twice_data=True,
    )
    train_unlabel_dataset = SEDDataset(
        train_unlabel_df,
        data_dir=(feat_dir / "train/unlabel_in_domain"),
        encode_function=encode_function,
        pooling_time_ratio=cfg["pooling_time_ratio"],
        transforms=train_transforms,
        twice_data=True,
    )

    valid_dataset = SEDDataset(
        valid_df,
        data_dir=(feat_dir / "validation"),
        encode_function=encode_function,
        pooling_time_ratio=cfg["pooling_time_ratio"],
        transforms=test_transforms,
    )

    if cfg["ngpu"] > 1:
        cfg["batch_size"] *= cfg["ngpu"]

    train_synth_loader = DataLoader(
        train_synth_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        drop_last=True,
        pin_memory=True,
    )
    train_weak_loader = DataLoader(
        train_weak_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        drop_last=True,
        pin_memory=True,
    )
    train_unlabel_loader = DataLoader(
        train_unlabel_dataset,
        batch_size=cfg["batch_size"] * 2,
        shuffle=True,
        num_workers=cfg["num_workers"],
        drop_last=True,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
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

    model = SEDModel(n_class=len(classes), cnn_kwargs=cfg["model"]["cnn"], encoder_kwargs=cfg["model"]["encoder"])
    ema_model = SEDModel(n_class=len(classes), cnn_kwargs=cfg["model"]["cnn"], encoder_kwargs=cfg["model"]["encoder"])

    # Show network architecture details
    logging.info(model)
    logging.info(model.parameters())
    logging.info(f"model parameter: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    wandb.watch(model)

    trainer_options = MeanTeacherTrainerOptions(**cfg["trainer_options"])
    trainer_options._set_validation_options(
        valid_meta=cfg["valid_meta"],
        valid_audio_dir=cfg["valid_audio_dir"],
        max_len_seconds=cfg["max_len_seconds"],
        sample_rate=cfg["feature"]["sample_rate"],
        hop_size=cfg["feature"]["mel_spec"]["hop_size"],
        pooling_time_ratio=cfg["pooling_time_ratio"],
    )

    # set optimizer and lr scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    if cfg["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(trainable_params, **cfg["optimizer_params"])
    else:
        import torch_optimizer as optim

        optimizer = getattr(optim, cfg["optimizer"])(trainable_params, **cfg["optimizer_params"])

    scheduler = getattr(torch.optim.lr_scheduler, cfg["scheduler"])(optimizer, **cfg["scheduler_params"])

    trainer = MeanTeacherTrainer(
        model=model,
        ema_model=ema_model,
        strong_loader=train_synth_loader,
        weak_loader=train_weak_loader,
        unlabel_loader=train_unlabel_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        exp_name=exp_name,
        pretrained=cfg["pretrained"],
        resume=cfg["resume"],
        trainer_options=trainer_options,
    )

    trainer.run()


if __name__ == "__main__":
    main(sys.argv[1:])
