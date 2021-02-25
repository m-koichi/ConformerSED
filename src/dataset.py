from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SEDDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        data_dir: Path,
        encode_function,
        pooling_time_ratio: int = 1,
        transforms=None,
        twice_data=False,
    ):
        self.df = df
        self.data_dir = data_dir
        self.encode_function = encode_function
        self.ptr = pooling_time_ratio
        self.transforms = transforms
        self.filenames = df.filename.drop_duplicates().values
        self.features = {}
        self.twice_data = twice_data

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        data_id = self.filenames[index]
        data = self._get_sample(data_id)
        label = self._get_label(data_id)
        if self.transforms is not None:
            data, label = self.transforms((data, label))

        # label pooling here because data augmentation may handle label (e.g. time shifting)
        # select center frame as a pooled label
        label = label[self.ptr // 2 :: self.ptr, :]

        # Return twice data with different augmentation if use mean teacher training
        if not self.twice_data:
            return (
                torch.from_numpy(data).float().unsqueeze(0),
                torch.from_numpy(label).float(),
                data_id,
            )
        else:
            return (
                torch.from_numpy(data[0]).float().unsqueeze(0),
                torch.from_numpy(data[1]).float().unsqueeze(0),
                torch.from_numpy(label).float(),
                data_id,
            )

    def _get_sample(self, filename):
        if self.features.get(filename) is None:
            data = np.load((self.data_dir / filename.replace("wav", "npy"))).astype(np.float32)
            self.features[filename] = data
        else:
            data = self.features[filename]
        return data

    def _get_label(self, filename):
        if "event_labels" in self.df.columns or {"onset", "offset", "event_label"}.issubset(self.df.columns):
            # get weak label
            if "event_labels" in self.df.columns:
                label = self.df[self.df.filename == filename]["event_labels"].values[0]
                if pd.isna(label):
                    label = []
                if type(label) is str:
                    if label == "":
                        label = []
                    else:
                        label = label.split(",")
            # get strong label
            else:
                cols = ["onset", "offset", "event_label"]
                label = self.df[self.df.filename == filename][cols]
                if label.empty:
                    label = []
        else:
            label = "empty"  # trick to have -1 for unlabeled data and concat them with labeled
            if "filename" not in self.df.columns:
                raise NotImplementedError(
                    "Dataframe to be encoded doesn't have specified columns: columns allowed: 'filename' for unlabeled;"
                    "'filename', 'event_labels' for weak labels; 'filename' 'onset' 'offset' 'event_label' "
                    "for strong labels, yours: {}".format(self.df.columns)
                )
        if self.encode_function is not None:
            # labels are a list of string or list of list [[label, onset, offset]]
            label = self.encode_function(label)
        return label
