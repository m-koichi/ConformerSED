import argparse
import logging
import subprocess
from pathlib import Path

import librosa
import numpy as np
import yaml
from joblib import Parallel, delayed


def resample_wav(cfg, src, dest):
    sox_option = f'--norm={cfg["gain"]} -r {cfg["sample_rate"]} -c 1 -t wav'
    filter_option = f'highpass {cfg["highpass"]}'
    subprocess.run(f"sox {src} {sox_option} {dest} {filter_option}", shell=True)


def feature_extraction(cfg, src, dest):
    y, sr = librosa.load(src, sr=None)
    mel_spec = calculate_mel_spec(y, sr, **cfg["mel_spec"])
    np.save(dest, mel_spec)


def calculate_mel_spec(x, fs, n_mels, n_fft, hop_size, fmin=None, fmax=None):
    fmin = 0 if fmin is None else fmin
    fmax = fs / 2 if fmax is None else fmax
    # Compute spectrogram
    ham_win = np.hamming(n_fft)

    spec = librosa.stft(x, n_fft=n_fft, hop_length=hop_size, window=ham_win, center=True, pad_mode="reflect")

    mel_spec = librosa.feature.melspectrogram(
        S=np.abs(spec),  # amplitude, for energy: spec**2 but don't forget to change amplitude_to_db.
        sr=fs,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        htk=False,
        norm=None,
    )

    # if self.save_log_feature:
    #     mel_spec = librosa.amplitude_to_db(mel_spec)  # 10 * log10(S**2 / ref), ref default is 1
    mel_spec = mel_spec.T
    mel_spec = mel_spec.astype(np.float32)
    return mel_spec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--nj", type=int, default=1)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)["feature"]

    wav_dir = Path(f"./data/wav/sr{cfg['sample_rate']}")
    if not wav_dir.exists():
        wav_dir.mkdir(parents=True)
        for x in ["train/synthetic20", "train/weak", "train/unlabel_in_domain", "validation", "eval", "public"]:
            src_dir = Path(cfg["audio_root"]) / x
            dest_dir = wav_dir / x
            dest_dir.mkdir(parents=True)
            Parallel(n_jobs=args.nj)(
                delayed(resample_wav)(cfg, filename, (dest_dir / filename.name)) for filename in src_dir.glob("*.wav")
            )
    else:
        logging.info(f"{wav_dir} is already exists, resampling is skipped.")

    feat_dir = Path(
        f"./data/feat/sr{cfg['sample_rate']}"
        + f"_n_mels{cfg['mel_spec']['n_mels']}_n_fft{cfg['mel_spec']['n_fft']}_hop_size{cfg['mel_spec']['hop_size']}"
    )
    if not feat_dir.exists():
        feat_dir.mkdir(parents=True)
        for x in ["train/synthetic20", "train/weak", "train/unlabel_in_domain", "validation", "eval", "public"]:
            src_dir = wav_dir / x
            dest_dir = feat_dir / x
            dest_dir.mkdir(parents=True)
            Parallel(n_jobs=args.nj)(
                delayed(feature_extraction)(cfg, filename, (dest_dir / filename.stem))
                for filename in src_dir.glob("*.wav")
            )
    else:
        logging.info(f"{feat_dir} is already exists, feature extraction is skipped.")


if __name__ == "__main__":
    main()
