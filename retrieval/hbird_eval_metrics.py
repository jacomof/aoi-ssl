# Adapted from: https://github.com/vpariza/open-hummingbird-eval/blob/main/hbird/utils/eval_metrics.py
# Adapted per-class IoU metric computation using hungarian matching
# (needed in multi-class classification as hummingbird is essentially a
# clustering methods and you don't know the id/pos of each class in the results).
# It turns out the matching does not seem to be necessary in our
# multi-label setting.

import torch
import numpy as np
from typing import List, Tuple, Dict
from joblib import Parallel, delayed
from scipy.optimize import linear_sum_assignment
from collections import defaultdict


class PredsmIoU:
    """
    Subclasses Metric. Computes mean Intersection over Union (mIoU) given ground-truth and predictions.
    .update() can be called repeatedly to add data from multiple validation loops.
    """

    def __init__(
        self,
        num_pred_classes: int,
        num_gt_classes: int,
        threshold: list = [0.5, 0.5, 0.5, 0.5],
        class_names=["wire", "ball", "wedge", "epoxy"],
    ):
        """
        :param num_pred_classes: The number of predicted classes.
        :param num_gt_classes: The number of gt classes.
        """
        self.num_pred_classes = num_pred_classes
        self.num_gt_classes = num_gt_classes
        self.gt = []
        self.pred = []
        self.n_jobs = -1
        self.threshold = threshold
        self.threshold = np.array(threshold, dtype=np.float32)
        self.class_names = class_names

    def update(self, gt: torch.Tensor, pred: torch.Tensor) -> None:
        self.gt.append(gt)
        self.pred.append(pred)

    def compute(
        self,
        is_global_zero: bool,
        many_to_one: bool = False,
        precision_based: bool = False,
        linear_probe: bool = False,
        thresholds: list = None,
    ) -> Tuple[
        float, List[np.int64], List[np.int64], List[np.int64], List[np.int64], float
    ]:
        """
        Compute mIoU with optional hungarian matching or many-to-one matching (extracts information from labels).
        :param is_global_zero: Flag indicating whether process is rank zero. Computation of metric is only triggered
        if True.
        :param many_to_one: Compute a many-to-one mapping of predicted classes to ground truth instead of hungarian
        matching.
        :param precision_based: Use precision as matching criteria instead of IoU for assigning predicted class to
        ground truth class.
        :param linear_probe: Skip hungarian / many-to-one matching. Used for evaluating predictions of fine-tuned heads.
        :return: mIoU over all classes, true positives per class, false negatives per class, false positives per class,
        reordered predictions matching gt,  percentage of clusters matched to background class. 1/self.num_pred_classes
        if self.num_pred_classes == self.num_gt_classes.
        """

        if is_global_zero:
            pred = torch.cat(self.pred).cpu().numpy()
            gt = torch.cat(self.gt).cpu().numpy().astype(int)  # Ensure gt is int
            return self.compute_miou(
                gt,
                pred,
                self.num_pred_classes,
                self.num_gt_classes,
                many_to_one=many_to_one,
                precision_based=precision_based,
                linear_probe=linear_probe,
            )

    def compute_miou(
        self,
        gt: np.ndarray,
        pred: np.ndarray,
        num_pred: int,
        num_gt: int,
        many_to_one=False,
        precision_based=False,
        linear_probe=False,
    ) -> Tuple[
        float, List[np.int64], List[np.int64], List[np.int64], List[np.int64], float
    ]:
        """
        Compute mIoU with optional hungarian matching or many-to-one matching (extracts information from labels).
        :param gt: numpy array with all flattened ground-truth class assignments per pixel
        :param pred: numpy array with all flattened class assignment predictions per pixel
        :param num_pred: number of predicted classes
        :param num_gt: number of ground truth classes
        :param many_to_one: Compute a many-to-one mapping of predicted classes to ground truth instead of hungarian
        matching.
        :param precision_based: Use precision as matching criteria instead of IoU for assigning predicted class to
        ground truth class.
        :param linear_probe: Skip hungarian / many-to-one matching. Used for evaluating predictions of fine-tuned heads.
        :return: mIoU over all classes, true positives per class, false negatives per class, false positives per class,
        reordered predictions matching gt,  percentage of clusters matched to background class. 1/self.num_pred_classes
        if self.num_pred_classes == self.num_gt_classes.
        """
        assert pred.shape == gt.shape
        tp = [0] * num_gt
        fp = [0] * num_gt
        fn = [0] * num_gt
        jac = [0] * num_gt

        if linear_probe:
            reordered_preds = pred
        else:
            match = self._hungarian_match(num_pred, num_gt, pred, gt)
            # remap predictions
            reordered_preds = np.zeros((len(pred), self.num_gt_classes))
            reordered_preds = pred.copy()
            for gt_idx, pred_idx in zip(match[0], match[1]):
                reordered_preds[..., gt_idx] = pred[..., pred_idx]

        # tp, fp, and fn evaluation
        # tp, fp, and fn evaluation
        for i_part in range(0, num_gt):
            # For multilabel, work with binary masks per class
            tmp_all_gt = (gt[..., i_part]).flatten()  # GT binary mask for class i_part
            tmp_pred = (
                reordered_preds[..., i_part] > self.threshold[i_part]
            ).flatten()  # Pred binary mask for class i_part
            if tmp_all_gt.sum() == 0 and tmp_pred.sum() == 0:
                # If both GT and prediction are empty, don't count this class
                jac[i_part] = None
            else:
                tp[i_part] += np.sum(tmp_all_gt & tmp_pred)
                fp[i_part] += np.sum(~tmp_all_gt & tmp_pred)
                fn[i_part] += np.sum(tmp_all_gt & ~tmp_pred)

        # Calculate IoU per class
        for i_part in range(0, num_gt):
            if jac[i_part] is not None:
                jac[i_part] = float(tp[i_part]) / max(
                    float(tp[i_part] + fp[i_part] + fn[i_part]), 1e-8
                )

        return jac, tp, fp, fn

    def compute_score_matrix(
        self,
        num_pred: int,
        num_gt: int,
        pred: np.ndarray,
        gt: np.ndarray,
        precision_based: bool = False,
    ) -> np.ndarray:
        """
        Compute score matrix. Each element i, j of matrix is the score if i was matched j. Computation is parallelized
        over self.n_jobs.
        :param num_pred: number of predicted classes
        :param num_gt: number of ground-truth classes
        :param pred: flattened predictions
        :param gt: flattened gt
        :param precision_based: flag to calculate precision instead of IoU.
        :return: num_pred x num_gt matrix with A[i, j] being the score if ground-truth class i was matched to
        predicted class j.
        """
        score_mat = Parallel(n_jobs=self.n_jobs)(
            delayed(self.get_score)(pred, gt, c1, c2, precision_based=precision_based)
            for c2 in range(num_pred)
            for c1 in range(num_gt)
        )
        score_mat = np.array(score_mat)
        return score_mat.reshape((num_pred, num_gt)).T

    def compute_binary_iou(
        self, gt_binary: np.ndarray, pred_binary: np.ndarray
    ) -> float:
        """
        Compute IoU for binary masks.
        :param gt_binary: Binary mask for ground truth class
        :param pred_binary: Binary mask for predicted class
        :return: IoU score
        """
        tp = np.sum(gt_binary & pred_binary)
        fp = np.sum(~gt_binary & pred_binary)
        fn = np.sum(gt_binary & ~pred_binary)
        return float(tp) / max(float(tp + fp + fn), 1e-8)

    def _hungarian_match(
        self, num_pred: int, num_gt: int, pred: np.ndarray, gt: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        score_matrix = np.zeros(
            (self.num_gt_classes, self.num_pred_classes), dtype=np.float32
        )
        # For each class independently
        for gt_class in range(self.num_gt_classes):
            for pred_class in range(self.num_pred_classes):
                # Treat as binary classification problem
                gt_binary = gt[:, :, :, gt_class]
                pred_binary = (
                    pred[:, :, :, pred_class] > self.threshold[pred_class]
                )  # Binary mask for this pred class

                # Compute IoU for this binary pair
                iou_score = self.compute_binary_iou(gt_binary, pred_binary)
                score_matrix[gt_class, pred_class] = iou_score

        match = linear_sum_assignment(1 - score_matrix)
        return match

    def _original_match(
        self, num_pred, num_gt, pred, gt, precision_based=False
    ) -> Dict[int, list]:
        score_mat = self.compute_score_matrix(
            num_pred, num_gt, pred, gt, precision_based=precision_based
        )
        preds_to_gts = {}
        preds_to_gt_scores = {}
        # Greedily match predicted class to ground-truth class by best score.
        for pred_c in range(num_pred):
            for gt_c in range(num_gt):
                score = score_mat[gt_c, pred_c]
                if (pred_c not in preds_to_gts) or (score > preds_to_gt_scores[pred_c]):
                    preds_to_gts[pred_c] = gt_c
                    preds_to_gt_scores[pred_c] = score
        gt_to_matches = defaultdict(list)
        for k, v in preds_to_gts.items():
            gt_to_matches[v].append(k)
        return gt_to_matches
