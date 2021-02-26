import random
import warnings

import librosa
import numpy as np


class Transform(object):
    def transform_data(self, data):
        # Mandatory to be defined by subclasses
        raise NotImplementedError("Abstract object")

    def transform_label(self, label):
        # Do nothing, to be changed in subclasses if needed
        return label

    def _apply_transform(self, sample_no_index):
        data, label = sample_no_index
        if type(data) is tuple:  # meaning there is more than one data_input (could be duet, triplet...)
            data = list(data)
            for k in range(len(data)):
                data[k] = self.transform_data(data[k])
            data = tuple(data)
        else:
            data = self.transform_data(data)
        label = self.transform_label(label)
        return data, label

    def __call__(self, sample):
        """Apply the transformation
        Args:
            sample: tuple, a sample defined by a DataLoad class

        Returns:
            tuple
            The transformed tuple
        """
        if type(sample[1]) is int:  # Means there is an index, may be another way to make it cleaner
            sample_data, index = sample
            sample_data = self._apply_transform(sample_data)
            sample = sample_data, index
        else:
            sample = self._apply_transform(sample)
        return sample


class RandomTransforms(object):
    """Base class for a list of transformations with randomness
    Args:
        transforms (list or tuple): list of transformations
    """

    def __init__(self, transforms):
        assert isinstance(transforms, (list, tuple))
        self.transforms = transforms

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class RandomApply(RandomTransforms):
    """Apply randomly a list of transformations with a given probability
    Args:
        transforms (list or tuple): list of transformations
        p (float): probability
    """

    def __init__(self, transforms, p=0.5):
        super(RandomApply, self).__init__(transforms)
        self.p = p

    def __call__(self, sample):
        if self.p < random.random():
            return sample
        for t in self.transforms:
            sample = t(sample)
        return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        format_string += "\n    p={}".format(self.p)
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Normalize(Transform):
    def __init__(self, mean=None, std=None, mode="gcmvn"):
        self.mean = mean
        self.std = std
        self.mode = mode
        self.ref_level_db = 20
        self.min_level_db = -80

    def transform_data(self, data):
        if self.mode == "gcmvn":
            return (data - self.mean) / self.std
        elif self.mode == "cmvn":
            return (data - data.mean(axis=0)) / data.std(axis=0)
        elif self.mode == "cmn":
            return data - data.mean(axis=0)
        elif self.mode == "min_max":
            data -= self.ref_level_db
            return np.clip((data - self.min_level_db) / -self.min_level_db, 0, 1)


class DataTwice(Transform):
    def __call__(self, sample):
        data, label = sample
        return (data, np.copy(data)), label


class AugmentGaussianNoise(Transform):
    """Pad or truncate a sequence given a number of frames
    Args:
        mean: float, mean of the Gaussian noise to add
    Attributes:
        std: float, std of the Gaussian noise to add
    """

    def __init__(self, mean=0.0, std=None, snr=None):
        self.mean = mean
        self.std = std
        self.snr = snr

    @staticmethod
    def gaussian_noise(features, snr):
        """Apply gaussian noise on each point of the data

        Args:
            features: numpy.array, features to be modified
        Returns:
            numpy.ndarray
            Modified features
        """
        # If using source separation, using only the first audio (the mixture) to compute the gaussian noise,
        # Otherwise it just removes the first axis if it was an extended one
        if len(features.shape) == 3:
            feat_used = features[0]
        else:
            feat_used = features
        std = np.sqrt(np.mean((feat_used ** 2) * (10 ** (-snr / 10)), axis=-2))
        try:
            noise = np.random.normal(0, std, features.shape)
        except Exception as e:
            warnings.warn(f"the computed noise did not work std: {std}, using 0.5 for std instead")
            noise = np.random.normal(0, 0.5, features.shape)

        return features + noise

    def transform_data(self, data):
        """Apply the transformation on data
        Args:
            data: np.array, the data to be modified

        Returns:
            (np.array, np.array)
            (original data, noisy_data (data + noise))
            Note: return 2 values! needed for mean teacher!
        """
        if self.std is not None:
            noisy_data = data + np.abs(np.random.normal(0, 0.5 ** 2, data.shape))
        elif self.snr is not None:
            noisy_data = self.gaussian_noise(data, self.snr)
        else:
            raise NotImplementedError("Only (mean, std) or snr can be given")
        # return data, noisy_data
        return noisy_data


class ApplyLog(Transform):
    def __init__(self, zero_db=False):
        self.zero_db = zero_db

    def transform_data(self, sample):
        if self.zero_db:
            return librosa.amplitude_to_db(sample, ref=np.max)
        else:
            return librosa.amplitude_to_db(sample)


class TimeMask(Transform):
    def __init__(self, num_masks=1, mask_param=100):
        self.num_masks = num_masks
        self.mask_param = mask_param

    def transform_data(self, data):
        tau = data.shape[0]
        for i in range(self.num_masks):
            t = int(np.random.uniform(low=0.0, high=self.mask_param))
            t0 = random.randint(0, tau - t)
            data[t0 : t0 + t, :] = 0
        return data


class FrequencyMask(Transform):
    def __init__(self, num_masks=1, mask_param=20):
        self.num_masks = num_masks
        self.mask_param = mask_param

    def transform_data(self, data):
        v = data.shape[1]
        for i in range(self.num_masks):
            f = int(np.random.uniform(low=0.0, high=self.mask_param))
            f0 = random.randint(0, v - f)
            data[:, f0 : f0 + f] = 0
        return data


class TimeShift(Transform):
    def __init__(self, mean=0, std=90):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        data, label = sample
        shift = int(np.random.normal(self.mean, self.std))

        if type(data) is tuple:  # meaning there is more than one data_input (could be duet, triplet...)
            data = list(data)
            for k in range(len(data)):
                data[k] = np.roll(data[k], shift, axis=0)
            data = tuple(data)
        else:
            data = np.roll(data, shift, axis=0)

        if len(label.shape) == 2:
            label = np.roll(label, shift, axis=0)  # strong label only

        sample = (data, label)
        return sample


class FrequencyShift(Transform):
    def __init__(self, mean=0, std=3):
        self.mean = mean
        self.std = std

    def transform_data(self, data):
        shift = int(np.random.normal(self.mean, self.std))
        data = np.roll(data, shift, axis=1)
        return data


class PadOrTrunc(Transform):
    """Pad or truncate a sequence given a number of frames
    Args:
        nb_frames: int, the number of frames to match
    Attributes:
        nb_frames: int, the number of frames to match
    """

    def __init__(self, nb_frames, apply_to_label=False):
        self.nb_frames = nb_frames
        self.apply_to_label = apply_to_label

    def transform_label(self, label):
        if self.apply_to_label:
            if label is not None:
                if label.shape == 2:
                    return pad_trunc_seq(label, self.nb_frames)  # strong label
                else:
                    return label  # weak label
        else:
            return label

    def transform_data(self, data):
        """Apply the transformation on data
        Args:
            data: np.array, the data to be modified

        Returns:
            np.array
            The transformed data
        """
        return pad_trunc_seq(data, self.nb_frames)


def pad_trunc_seq(x, max_len):
    """Pad or truncate a sequence data to a fixed length.
    The sequence should be on axis -2.

    Args:
      x: ndarray, input sequence data.
      max_len: integer, length of sequence to be padded or truncated.

    Returns:
      ndarray, Padded or truncated input sequence data.
    """
    shape = x.shape
    if shape[-2] <= max_len:
        padded = max_len - shape[-2]
        padded_shape = ((0, 0),) * len(shape[:-2]) + ((0, padded), (0, 0))
        x = np.pad(x, padded_shape, mode="constant")
    else:
        x = x[..., :max_len, :]
    return x


class Compose(object):
    """Composes several transforms together.
    Args:
        transforms: list of ``Transform`` objects, list of transforms to compose.
        Example of transform: ToTensor()
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def add_transform(self, transform):
        t = self.transforms.copy()
        t.append(transform)
        return Compose(t)

    def __call__(self, audio):
        for t in self.transforms:
            audio = t(audio)
        return audio

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"

        return format_string


def get_transforms(cfg, nb_frames, norm_dict_params=None, training=True, prob=1.0):

    transf = []
    if cfg["semi_supervised_training"] and training:
        transf.append(DataTwice())
    if cfg["time_shift"]["apply"] and training:
        transf.append(RandomApply([TimeShift(**cfg["time_shift"]["params"])], p=prob))
    if cfg["frequency_shift"]["apply"] and training:
        transf.append(RandomApply([FrequencyShift(**cfg["frequency_shift"]["params"])], p=prob))
    if cfg["add_noise"]["apply"] is not None and training:
        transf.append(RandomApply([AugmentGaussianNoise(**cfg["add_noise"]["params"])], p=prob))
    transf.append(ApplyLog())
    transf.append(Normalize(**norm_dict_params))
    if cfg["frequency_mask"]["apply"] and training:
        transf.append(RandomApply([FrequencyMask(**cfg["frequency_mask"]["params"])], p=prob))
    transf.append(PadOrTrunc(nb_frames=nb_frames, apply_to_label=True))

    return Compose(transf)
