import functools
import logging

import h5py
import numpy as np
import pandas as pd
import torch
from dcase_util.data import ProbabilityEncoder
from genericpath import exists
from scipy.signal import medfilt
from torch.utils.data import DataLoader, Dataset

from evaluation_measures import ConfusionMatrix, compute_metrics, compute_psds_from_operating_points, psds_score


def median_filt_1d(event_roll, filt_span=7):
    """FUNCTION TO APPLY MEDIAN FILTER
    ARGS:
    --
    event_roll: event roll [T,C]
    filt_span: median filter span(integer odd scalar)
    RETURN:
    --
    event_roll : median filter applied event roll [T,C]
    """
    assert isinstance(filt_span, (int, list))
    if len(event_roll.shape) == 1:
        event_roll = medfilt(event_roll, filt_span)
    else:
        if isinstance(filt_span, int):
            for i in range(event_roll.shape[1]):
                event_roll[:, i] = medfilt(event_roll[:, i], filt_span)
        else:
            assert event_roll.shape[1] == len(filt_span)
            for i in range(event_roll.shape[1]):
                event_roll[:, i] = medfilt(event_roll[:, i], filt_span[i])

    return event_roll


def fill_up_gap(event_roll, accept_gap=5):
    """FUNCTION TO PERFORM FILL UP GAPS
    ARGS:
    --
    event_roll: event roll [T,C]
    accept_gap: number of accept gap to fill up (integer scalar)
    RETURN:
    --
    event_roll: processed event roll [T,C]
    """
    assert isinstance(accept_gap, (int, list))
    num_classes = event_roll.shape[1]
    event_roll_ = np.append(
        np.append(np.zeros((1, num_classes)), event_roll, axis=0),
        np.zeros((1, num_classes)),
        axis=0,
    )
    aux_event_roll = np.diff(event_roll_, axis=0)

    for i in range(event_roll.shape[1]):
        onsets = np.where(aux_event_roll[:, i] == 1)[0]
        offsets = np.where(aux_event_roll[:, i] == -1)[0]
        for j in range(1, onsets.shape[0]):
            if isinstance(accept_gap, int):
                if onsets[j] - offsets[j - 1] <= accept_gap:
                    event_roll[offsets[j - 1] : onsets[j], i] = 1
            elif isinstance(accept_gap, list):
                if onsets[j] - offsets[j - 1] <= accept_gap[i]:
                    event_roll[offsets[j - 1] : onsets[j], i] = 1

    return event_roll


def remove_short_duration(event_roll, reject_duration=10):
    """Remove short duration
    ARGS:
    --
    event_roll: event roll [T,C]
    reject_duration: number of duration to reject as short section (int or list)
    RETURN:
    --
    event_roll: processed event roll [T,C]
    """
    assert isinstance(reject_duration, (int, list))
    num_classes = event_roll.shape[1]
    event_roll_ = np.append(
        np.append(np.zeros((1, num_classes)), event_roll, axis=0),
        np.zeros((1, num_classes)),
        axis=0,
    )
    aux_event_roll = np.diff(event_roll_, axis=0)

    for i in range(event_roll.shape[1]):
        onsets = np.where(aux_event_roll[:, i] == 1)[0]
        offsets = np.where(aux_event_roll[:, i] == -1)[0]
        for j in range(onsets.shape[0]):
            if isinstance(reject_duration, int):
                if onsets[j] - offsets[j] <= reject_duration:
                    event_roll[offsets[j] : onsets[j], i] = 0
            elif isinstance(reject_duration, list):
                if onsets[j] - offsets[j] <= reject_duration[i]:
                    event_roll[offsets[j] : onsets[j], i] = 0

    return event_roll


class ScoreDataset(Dataset):
    def __init__(self, score_h5_path, has_label=True):
        with h5py.File(score_h5_path, "r") as h5:
            self.data_ids = list(h5.keys())
            self.dataset = {}
            self.has_label = has_label
            for data_id in self.data_ids:
                pred_strong = h5[data_id]["pred_strong"][()]
                pred_weak = h5[data_id]["pred_weak"][()]
                if self.has_label:
                    target = h5[data_id]["target"][()]
                    self.dataset[data_id] = dict(pred_strong=pred_strong, pred_weak=pred_weak, target=target)
                else:
                    self.dataset[data_id] = dict(
                        pred_strong=pred_strong,
                        pred_weak=pred_weak,
                    )

    def __getitem__(self, index):
        data_id = self.data_ids[index]
        pred_strong = self.dataset[data_id]["pred_strong"]
        pred_weak = self.dataset[data_id]["pred_weak"]
        if self.has_label:
            target = self.dataset[data_id]["target"]
            return dict(
                data_id=data_id,
                pred_strong=pred_strong,
                pred_weak=pred_weak,
                target=target,
            )
        else:
            return dict(
                data_id=data_id,
                pred_strong=pred_strong,
                pred_weak=pred_weak,
            )

    def __len__(self):
        return len(self.data_ids)


class PostProcess:
    def __init__(
        self,
        model: torch.nn.Module,
        iterator,
        output_dir,
        options,
    ):
        self.model = model
        self.iterator = iterator
        self.options = options
        self.device = options.device
        self.decoder = options.decoder
        self.pooling_time_ratio = options.pooling_time_ratio
        self.sample_rate = options.sample_rate
        self.hop_size = options.hop_size
        self.thresholds = [0.5]
        self.validation_df = options.validation_df
        self.durations_validation = options.durations_validation
        self.labels = {key: value for value, key in enumerate(options.classes)}
        self.output_dir = output_dir
        self.get_posterior(save_h5_path=output_dir / "posterior.h5")
        self.data_loader = DataLoader(ScoreDataset(output_dir / "posterior.h5", has_label=True))

    @torch.no_grad()
    def get_posterior(self, save_h5_path) -> None:
        with h5py.File(save_h5_path, "w") as h5:
            self.model.eval()
            for (batch_input, batch_target, data_ids) in self.iterator:
                predicts = self.model(batch_input.to(self.device))
                predicts["strong"] = torch.sigmoid(predicts["strong"]).cpu().data.numpy()
                predicts["weak"] = torch.sigmoid(predicts["weak"]).cpu().data.numpy()
                for data_id, pred_strong, pred_weak, target in zip(
                    data_ids, predicts["strong"], predicts["weak"], batch_target.numpy()
                ):
                    h5.create_group(data_id)
                    h5[data_id].create_dataset("pred_strong", data=pred_strong)
                    h5[data_id].create_dataset("pred_weak", data=pred_weak)
                    h5[data_id].create_dataset("target", data=target)

    def get_prediction_dataframe(
        self,
        post_processing=None,
        save_predictions=None,
        transforms=None,
        mode="validation",
        threshold=0.5,
        binarization_type="global_threshold",
    ):
        """
        post_processing: e.g. [functools.partial(median_filt_1d, filt_span=39)]
        """
        prediction_df = pd.DataFrame()

        # Flame level
        frame_measure = [ConfusionMatrix() for i in range(len(self.labels))]
        tag_measure = ConfusionMatrix()

        for batch_idx, data in enumerate(self.data_loader):
            output = {}
            output["strong"] = data["pred_strong"].cpu().data.numpy()
            output["weak"] = data["pred_weak"].cpu().data.numpy()

            # Binarize score into predicted label
            if binarization_type == "class_threshold":
                for i in range(output["strong"].shape[0]):
                    output["strong"][i] = ProbabilityEncoder().binarization(
                        output["strong"][i],
                        binarization_type=binarization_type,
                        threshold=threshold,
                        time_axis=0,
                    )
            elif binarization_type == "global_threshold":
                output["strong"] = ProbabilityEncoder().binarization(
                    output["strong"],
                    binarization_type=binarization_type,
                    threshold=threshold,
                )
            else:
                raise ValueError("binarization_type must be 'class_threshold' or 'global_threshold'")
            weak = ProbabilityEncoder().binarization(
                output["weak"], binarization_type="global_threshold", threshold=0.5
            )

            for pred, data_id in zip(output["strong"], data["data_id"]):
                # Apply post processing if exists
                if post_processing is not None:
                    for post_process_fn in post_processing:
                        pred = post_process_fn(pred)

                pred = self.decoder(pred)
                pred = pd.DataFrame(pred, columns=["event_label", "onset", "offset"])
                # Put them in seconds
                pred.loc[:, ["onset", "offset"]] *= self.pooling_time_ratio / (self.sample_rate / self.hop_size)
                pred.loc[:, ["onset", "offset"]] = pred[["onset", "offset"]].clip(0, self.options.max_len_seconds)
                pred["filename"] = data_id
                prediction_df = prediction_df.append(pred, ignore_index=True)

        return prediction_df

    def search_best_threshold(self, step, target="Event"):
        assert 0 < step < 1.0
        assert target in ["Event", "Frame"]
        best_th = {k: 0.0 for k in self.labels}
        best_f1 = {k: 0.0 for k in self.labels}

        for th in np.arange(step, 1.0, step):
            logging.info(f"threshold: {th}")
            prediction_df = self.get_prediction_dataframe(
                threshold=th,
                binarization_type="global_threshold",
                save_predictions=None,
            )
            events_metric, segments_metric, psds_m_f1 = compute_metrics(
                prediction_df, self.validation_df, self.durations_validation
            )

            for i, label in enumerate(self.labels):
                f1 = events_metric.class_wise_f_measure(event_label=label)["f_measure"]
                # if target == 'Event':
                #     f1 = valid_events_metric.class_wise_f_measure(event_label=label)['f_measure']
                # elif target == 'Frame':
                #     f1 = frame_measure[i].calc_f1()[2]
                # else:
                #     raise NotImplementedError
                if f1 > best_f1[label]:
                    best_th[label] = th
                    best_f1[label] = f1

        thres_list = [0.5] * len(self.labels)
        for i, label in enumerate(self.labels):
            thres_list[i] = best_th[label]

        prediction_df = self.get_prediction_dataframe(
            post_processing=None,
            threshold=thres_list,
            binarization_type="class_threshold",
        )

        # Compute evaluation metrics
        events_metric, segments_metric, psds_m_f1 = compute_metrics(
            prediction_df, self.validation_df, self.durations_validation
        )
        macro_f1_event = events_metric.results_class_wise_average_metrics()["f_measure"]["f_measure"]
        macro_f1_segment = segments_metric.results_class_wise_average_metrics()["f_measure"]["f_measure"]

        logging.info(f"Event-based F1:{macro_f1_event * 100:.4}\tSegment-based F1:{macro_f1_event * 100:.4}")
        logging.info(f"best_th: {best_th}")
        logging.info(f"best_f1: {best_f1}")
        return best_th, best_f1

    def search_best_median(self, spans, best_th=None, target="Event"):
        best_span = {k: 1 for k in self.labels}
        best_f1 = {k: 0.0 for k in self.labels}

        for span in spans:
            logging.info(f"median filter span: {span}")
            post_process_fn = [functools.partial(median_filt_1d, filt_span=span)]
            if best_th is not None:
                prediction_df = self.get_prediction_dataframe(
                    post_processing=post_process_fn,
                    threshold=list(best_th.values()),
                    binarization_type="class_threshold",
                )
            else:
                prediction_df = self.get_prediction_dataframe(post_processing=post_process_fn)
            events_metric, segments_metric, psds_m_f1 = compute_metrics(
                prediction_df, self.validation_df, self.durations_validation
            )
            for i, label in enumerate(self.labels):
                f1 = events_metric.class_wise_f_measure(event_label=label)["f_measure"]
                # if target == 'Event':
                #     f1 = valid_events_metric.class_wise_f_measure(event_label=label)['f_measure']
                # elif target == 'Frame':
                #     f1 = frame_measure[i].calc_f1()[2]
                # else:
                #     raise NotImplementedError
                if f1 > best_f1[label]:
                    best_span[label] = span
                    best_f1[label] = f1

        post_process_fn = [functools.partial(median_filt_1d, filt_span=list(best_span.values()))]
        if best_th is not None:
            prediction_df = self.get_prediction_dataframe(
                post_processing=post_process_fn,
                threshold=len(best_th.values()),
                binarization_type="class_threshold",
            )
        else:
            prediction_df = self.get_prediction_dataframe(post_processing=post_process_fn)
        # Compute evaluation metrics
        events_metric, segments_metric, psds_m_f1 = compute_metrics(
            prediction_df, self.validation_df, self.durations_validation
        )
        macro_f1_event = events_metric.results_class_wise_average_metrics()["f_measure"]["f_measure"]
        macro_f1_segment = segments_metric.results_class_wise_average_metrics()["f_measure"]["f_measure"]

        logging.info(f"best_span: {best_span}")
        logging.info(f"best_f1: {best_f1}")
        return best_span, best_f1

    def show_best(self, pp_params, save_predictions=None):
        # Set applying post-processing functions
        post_processing_fn = []
        if "threshold" in pp_params.keys():
            threshold = list(pp_params["threshold"].values())
            binarization_type = "class_threshold"
        else:
            threshold = 0.5
            binarization_type = "global_threshold"
        if "median_filtering" in pp_params.keys():
            filt_span = list(pp_params["median_filtering"].values())
            post_processing_fn.append(functools.partial(median_filt_1d, filt_span=filt_span))
        if "fill_up_gap" in pp_params.keys():
            accept_gap = list(pp_params["fill_up_gap"].values())
            post_processing_fn.append(functools.partial(fill_up_gap, accept_gap=accept_gap))
        if len(post_processing_fn) == 0:
            post_processing_fn = None

        prediction_df = self.get_prediction_dataframe(
            post_processing=post_processing_fn,
            threshold=threshold,
            binarization_type=binarization_type,
        )

        # Compute evaluation metrics
        events_metric, segments_metric, psds_m_f1 = compute_metrics(
            prediction_df, self.validation_df, self.durations_validation
        )
        macro_f1_event = events_metric.results_class_wise_average_metrics()["f_measure"]["f_measure"]
        macro_f1_segment = segments_metric.results_class_wise_average_metrics()["f_measure"]["f_measure"]

        logging.info(f"Event-based macro F1: {macro_f1_event}")
        logging.info(f"Segment-based macro F1: {macro_f1_segment}")

    def compute_psds(self):
        logging.info("Compute psds scores")
        ##########
        # Optional but recommended
        ##########
        # Compute psds scores with multiple thresholds (more accurate). n_thresholds could be increased.
        n_thresholds = 50
        out_nb_frames_1s = self.sample_rate / self.hop_size / self.pooling_time_ratio
        # median_window = max(int(0.45 * out_nb_frames_1s), 1)
        post_processing_fn = [functools.partial(median_filt_1d, filt_span=3)]

        # Example of 5 thresholds: 0.1, 0.3, 0.5, 0.7, 0.9
        list_thresholds = np.arange(1 / (n_thresholds * 2), 1, 1 / n_thresholds)

        prediction_dfs = {}
        for threshold in list_thresholds:
            prediction_dfs[threshold] = self.get_prediction_dataframe(
                post_processing=post_processing_fn,
                threshold=threshold,
                binarization_type="global_threshold",
            )
        pred_thresh = []
        for key in prediction_dfs:
            pred_thresh.append(prediction_dfs[key])

        if len(pred_thresh) == 1:
            pred_thresh = pred_thresh[0]

        # save predictions
        (self.output_dir / "predictions_thresh").mkdir(exist_ok=True)
        for th, pred_df in zip(list_thresholds, pred_thresh):
            pred_df.to_csv(
                self.output_dir / "predictions_thresh" / f"{th}.csv",
                index=False,
                sep="\t",
                float_format="%.3f",
            )

        psds = compute_psds_from_operating_points(pred_thresh, self.validation_df, self.durations_validation)
        psds_score(psds, filename_roc_curves=self.output_dir / "psds_roc.png")

    def tune_all(
        self,
    ):
        best_th, best_f1 = self.search_best_threshold(
            step=0.1,
        )

        best_fs, best_f1 = self.search_best_median(spans=list(range(1, 31, 2)), best_th=best_th)

        pp_params = {
            "threshold": best_th,
            "median_filtering": best_fs,
        }
        self.show_best(
            pp_params=pp_params,
        )

        logging.info("===================")
        logging.info(f"best_th: {best_th}")
        logging.info(f"best_fs: {best_fs}")
        return pp_params
