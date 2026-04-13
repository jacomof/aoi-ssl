# Image level retrieval for segmentation using an image encoder model.

import torch
from data.retrieval_dataset import RetrievalDataset
import os
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
from torchmetrics import JaccardIndex
from torchmetrics.wrappers import ClasswiseWrapper
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform
import cv2
import psutil
import time


class AddZeroChannel(ImageOnlyTransform):
    """Add a channel of all zeros to the image."""

    def __init__(self, always_apply=False, p=1.0):
        super(AddZeroChannel, self).__init__(always_apply, p)

    def apply(self, img, **params):
        # img shape: (H, W, C)
        h, w, c = img.shape
        # Create zero channel
        zero_channel = np.zeros((h, w, 1), dtype=img.dtype)
        # Concatenate along channel dimension
        return np.concatenate([img, zero_channel], axis=2)

    def get_transform_init_args_names(self):
        return ()


class CopyChannel0(ImageOnlyTransform):
    """Copy the first channel to the second channel."""

    def __init__(self, always_apply=False, p=1.0):
        super(CopyChannel0, self).__init__(always_apply, p)

    def apply(self, img, **params):
        # img shape: (H, W, C)
        return np.concatenate(
            [img[..., :1], img[..., :1], img[..., :1]], axis=-1
        )  # Copy first channel three times

    def get_transform_init_args_names(self):
        return ()


class SafeJaccardIndex:
    def __init__(self, jaccard_metric):
        self.jaccard_metric = jaccard_metric

    def __call__(self, preds, target):
        # For multi-label case, check each class individually
        result = {}
        for i, key in enumerate(self.jaccard_metric.labels):
            # Sum over all dimensions except the class dimension
            # target shape: (batch, classes, height, width)
            class_positive_count = target[:, i, :, :].sum()

            prefixed_key = f"iou_{key}"

            if class_positive_count == 0:
                result[prefixed_key] = None
            else:
                # Extract single class data and compute IoU
                class_preds = preds[:, i : i + 1, :, :]  # Keep dimension
                class_target = target[:, i : i + 1, :, :]  # Keep dimension

                # Create a binary metric for this single class
                from torchmetrics import JaccardIndex

                single_class_metric = JaccardIndex(task="binary")
                result[prefixed_key] = single_class_metric(
                    class_preds.squeeze(1), class_target.squeeze(1)
                ).item()

        return result


class KNNSegmentation:

    def __init__(
        self,
        encoder,
        classes=["wire", "ball", "wedge", "epoxy"],
        profile_time=False,
        train_loader=None,
        val_loader=None,
        config=None,
        normalize=False,
        train_im_list=None,
        val_im_list=None,
        batch_size=32,
        input_resolution=(512, 512),
        profile_memory=False,
    ):

        self.encoder = encoder
        self.device = next(encoder.parameters()).device
        self.representations = None
        self.distances = None
        self.labels = None
        self.classes = classes
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.profile_time = profile_time
        self.batch_size = batch_size
        self.input_resolution = input_resolution
        self.profile_memory = profile_memory

        if config is not None:
            self.classes = config.classes
            self.batch_size = config.batch_size
            self.data_path = Path(config.data_path)
            self.input_resolution = config.input_resolution
            self.profile_time = config.profile_time

            self.transform = self.create_transforms(
                normalize=normalize,
                pretrained_model=False,
            )

            # Transformations for random cropping evaluations
            self.transform2 = A.Compose(
                [
                    A.CropNonEmptyMaskIfExists(
                        self.input_resolution[0], self.input_resolution[1]
                    )
                ]
            )

            self.create_loaders(config)
        elif train_im_list is not None and val_im_list is not None:
            self.train_loader = self.create_loaders_from_list(
                train_im_list, self.batch_size, shuffle=False
            )
            self.val_loader = self.create_loaders_from_list(
                val_im_list, batch_size=self.batch_size, shuffle=False
            )

        # Update your __init__ method:
        self.iou_classwise = SafeJaccardIndex(
            ClasswiseWrapper(
                JaccardIndex(
                    task="multilabel",
                    num_labels=len(self.classes),
                    average="none",
                ),
                labels=self.classes,
                prefix="iou_",
            )
        )

    def create_loaders(self, config, train_set_size=0.7):
        """Create DataLoaders for training, validation, and testing.

        This is only used with when a config object is provided.

        Args:
            config (Namespace): Configuration object containing parameters.
            train_set_size (float): Proportion of the dataset to use for training (between 0 and 1).
        Returns:
            None
        """

        im_list = [
            self.data_path / "train" / img_file
            for img_file in os.listdir(self.data_path / "train")
        ]

        # Set random seed for reproducible splits
        self.train_indices = np.random.choice(
            len(im_list), int(train_set_size * len(im_list)), replace=False
        )

        self.val_indices = np.array(
            list(set(range(len(im_list))) - set(self.train_indices))
        )

        im_list = np.array(im_list)

        # Create separate image lists
        train_im_list = im_list[self.train_indices].tolist()
        val_im_list = im_list[self.val_indices].tolist()

        # Use ONLY training images for training dataset
        self.aoi_train = RetrievalDataset(
            train_im_list,  # Only training images!
            self.classes,
            transform=self.transform,
        )

        self.aoi_val = RetrievalDataset(
            val_im_list,  # Only validation images
            self.classes,
            transform=self.transform,
        )

        self.train_loader = DataLoader(
            self.aoi_train,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=4,
        )

        self.val_loader = DataLoader(
            self.aoi_val,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=4,
        )

        # Test loaders
        im_list_test = [
            self.data_path / "test" / img_file
            for img_file in os.listdir(self.data_path / "test")
        ]

        self.aoi_test = RetrievalDataset(
            im_list_test,
            self.classes,
            transform=self.transform,
        )

        self.test_loader = DataLoader(
            self.aoi_test,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=4,
        )

    def create_loaders_from_list(self, im_list, batch_size=32, shuffle=False):
        """Create DataLoaders from a list of image file paths.

        Args:
            im_list (list of str or Path): List of image file paths.
            batch_size (int): Batch size for the DataLoader.
            shuffle (bool): Whether to shuffle the data.
        Returns:
            DataLoader: DataLoader for the given image list.
        """

        transforms = self.create_transforms(normalize=False, pretrained_model=False)

        dataset = RetrievalDataset(
            im_list,
            self.classes,
            transform=transforms,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=4,
        )

    def create_transforms(self, normalize=False, pretrained_model=False):
        # Channel 0 mean: 0.17811504349642812, std: 0.22236312176409964
        # Channel 1 mean: 0.23459850405723615, std: 0.27069713773969
        normalization = A.Normalize(
            mean=[0.1781, 0.2224],  # Pretrain dataset mean
            std=[0.2346, 0.2707],  # Pretrain dataset std
        )

        transform_list = [
            A.PadIfNeeded(
                min_height=self.input_resolution[0],
                min_width=self.input_resolution[1],
                # Avoids reflective padding
                border_mode=cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
                p=1,
            ),
            A.CenterCrop(
                self.input_resolution[0],
                self.input_resolution[1],
            ),
        ]

        if pretrained_model:
            # Add dummy channel so RGB pretrained models can be used
            transform_list.append(AddZeroChannel(p=1.0))

        if normalize:
            transform_list.append(normalization)

        return A.Compose(transform_list)

    def l2_distance(self, test_reps: torch.Tensor, database_reps: torch.Tensor):
        """
        Compute the L2 distance between two tensors.
        """
        return torch.linalg.norm(
            (test_reps.unsqueeze(1) - database_reps.unsqueeze(0)), dim=2
        )

    def cosine_distance(self, test_reps: torch.Tensor, database_reps: torch.Tensor):
        """Compute the Cosine distance between two tensors.

        Cosine distance is defined as 1 - cosine similarity.

        Args:
            test_reps (torch.Tensor): Tensor of shape (N, D) where N is the number of test samples and D is the feature dimension.
            database_reps (torch.Tensor): Tensor of shape (M, D) where M is the number of database samples and D is the feature dimension.
        Returns:
            torch.Tensor: Tensor of shape (N, M) containing the cosine distances between each pair
        """
        test_reps = test_reps / (1e-8 + torch.norm(test_reps, dim=-1, keepdim=True))
        database_reps = database_reps / (
            1e-8 + torch.norm(database_reps, dim=-1, keepdim=True)
        )

        return 1 - torch.einsum(
            "ik,jk->ij", test_reps, database_reps
        )  # Cosine distance is 1 - cosine similarity

    def combine_labels(self, label_masks, threshold=0.5, distances=None):
        """Combine the labels of the K nearest neighbors using a distance-weighted average and thresholding.

        Args:
            label_masks (torch.Tensor): Tensor of shape (B, H, W, K, C) containing the labels of the K nearest neighbors for each pixel.
            threshold (float): Threshold for converting the averaged labels to binary.
            distances (torch.Tensor, optional): Tensor of shape (B, K) containing the distances of the K nearest neighbors. If provided, will be used for distance-weighted averaging.
        Returns:
            torch.Tensor: Tensor of shape (B, H, W, C) containing the combined binary labels for each pixel.

        """

        # label_masks: (B, H, W, K, C)

        if distances is not None:
            weights = 1 / (distances + 1e-8)  # Avoid division by zero
            weights = weights / weights.sum(dim=1, keepdim=True)  # Normalize weights
            weights = weights.view(
                weights.shape[0], 1, 1, weights.shape[1], 1
            )  # Reshape to (B, 1, 1, K, 1)
            label_masks = label_masks * weights  # Weight the labels by distances

        mean_labels = label_masks.sum(dim=3)  # Average over the K nearest neighbors
        mean_labels_thresholded = (
            mean_labels > threshold
        ).float()  # Threshold to get binary labels
        # (B, H, W, C)
        return mean_labels_thresholded

    def combine_labels_multi_thresholds(
        self, label_masks, thresholds=[0.5] * 4, distances=None
    ):
        """Combine the labels of the K nearest neighbors using a distance-weighted average and per-class thresholds.

        Args:
            label_masks (torch.Tensor): Tensor of shape (B, H, W, K, C) containing the labels of the K nearest neighbors for each pixel.
            thresholds (list of float): List of thresholds for each class for converting the averaged labels to binary.
            distances (torch.Tensor, optional): Tensor of shape (B, K) containing the distances of the K nearest neighbors. If provided, will be used for distance-weighted averaging.
        Returns:
            torch.Tensor: Tensor of shape (B, H, W, C) containing the combined binary labels for each pixel.
        """

        # label_masks: (B, H, W, K, C)

        if distances is not None:
            weights = 1 / (distances + 1e-8)  # Avoid division by zero
            weights = weights / weights.sum(dim=1, keepdim=True)  # Normalize weights
            weights = weights.view(
                weights.shape[0], 1, 1, weights.shape[1], 1
            )  # Reshape to (B, 1, 1, K, 1)
            label_masks = label_masks * weights  # Weight the labels by distances

        mean_labels = label_masks.sum(dim=3)  # Average over the K nearest neighbors
        thresholds = torch.tensor(thresholds, device=mean_labels.device).view(
            1, 1, 1, -1
        )  # Shape: (1, 1, 1, C)
        mean_labels_thresholded = (
            mean_labels > thresholds
        ).float()  # Threshold to get binary labels
        # (B, H, W, C)
        return mean_labels_thresholded

    def nearest_neighbors(
        self, test_representations: torch.Tensor, k, distance_metric="l2"
    ):
        """Find the k nearest neighbors in the database for each test representation.

        Args:
            test_representations (torch.Tensor): Tensor of shape (N, D) where N is the number of test samples and D is the feature dimension.
            k (int): Number of nearest neighbors to find.
            distance_metric (str): Distance metric to use ("l2" or "cosine").
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing:
                - distances (torch.Tensor): Tensor of shape (N, k) containing the distances to the k nearest neighbors.
                - indices (torch.Tensor): Tensor of shape (N, k) containing the indices of the k nearest neighbors in the database.
        """
        distances = None
        if distance_metric == "l2":
            distances = self.l2_distance(test_representations, self.representations)
        elif distance_metric == "cosine":
            distances = self.cosine_distance(test_representations, self.representations)

        top_dist, top_indices = torch.topk(
            distances, k=k, dim=1, largest=False, sorted=True
        )

        return top_dist, top_indices

    def compute_representations(self, dataloader):
        """
        Compute the representations of the training data using the encoder.
        """
        self.encoder.eval()
        cnt = 0
        total_images_loaded = 0
        first = True
        labels = []
        with torch.no_grad():
            for batch in dataloader:

                im = batch["image"]
                curr_mem = torch.cuda.memory_allocated()
                if self.profile_memory:
                    print(
                        f"Current CUDA memory allocated after loading image batch: {curr_mem / 1024**2:.2f} MB"
                    )

                im = im.to(next(self.encoder.parameters()).device)

                curr_mem = torch.cuda.memory_allocated()

                if self.profile_memory:
                    print(
                        f"Current CUDA memory allocated after moving image batch to cuda: {curr_mem / 1024**2:.2f} MB"
                    )

                representation = self.encoder(im)["x_norm_clstoken"]

                if first:
                    representations = representation
                    first = False
                else:
                    representations = torch.cat(
                        (representations, representation), dim=0
                    )

                labels.append(batch["class_mask"])

                if self.profile_memory:
                    print(
                        f"Current CUDA memory allocated after processing image batch: {curr_mem / 1024**2:.2f} MB"
                    )

                cnt += 1
                total_images_loaded += im.shape[0]

                if self.profile_memory:
                    print(f"Current RAM usage: {psutil.virtual_memory().percent}%")
                torch.cuda.empty_cache()

        self.representations = representations
        self.labels = torch.cat(labels, dim=0)

    def predict(
        self,
        x_test: torch.Tensor,
        k,
        distance_metric="cosine",
        weights="distance",
        threshold=0.5,
    ):
        """Predict the segmentation masks for the x_test images using KNN.

        Args:
            x_test (torch.Tensor): Tensor of shape (B, C, H, W) containing the test images.
            k (int): Number of nearest neighbors to use.
            distance_metric (str): Distance metric to use ("l2" or "cosine").
            weights (str): Weighting scheme to use ("distance" or "uniform").
            threshold (float or list of float): Threshold(s) for converting the averaged labels to binary.
        Returns:
            torch.Tensor: Tensor of shape (B, C, H, W) containing the predicted segmentation masks.
        """

        self.encoder.eval()
        if self.representations is None or self.labels is None:
            self.compute_representations(self.train_loader)

        with torch.no_grad():
            start_time = time.time()
            x_test = x_test.to(self.representations.device)
            end_time = time.time()
            if self.profile_time:
                print(
                    f"Time taken to move test data to device: {end_time - start_time:.2f} seconds"
                )

            start_time = time.time()
            x_test_representation = self.encoder(x_test)["x_norm_clstoken"]
            end_time = time.time()
            if self.profile_time:
                print(
                    f"Time taken to compute representation: {end_time - start_time:.2f} seconds"
                )

            start_time = time.time()
            top_distances, top_indices = self.nearest_neighbors(
                x_test_representation, k, distance_metric=distance_metric
            )

            # Shape of top_distances, top_indices: (batch_size, k)
            top_indices = top_indices.to(self.labels.device)
            top_distances = top_distances.to(self.labels.device)
            winner_labels = self.labels[top_indices].permute(
                0, 2, 3, 1, 4
            )  # Shape: (batch_size, k, H, W, C)
            end_time = time.time()
            if self.profile_time:
                print(
                    f"Time taken to compute nearest neighbors: {end_time - start_time:.2f} seconds"
                )

            distances = None
            if weights == "distance":
                distances = top_distances

            start_time = time.time()
            if type(threshold) is list:
                # If thresholds are provided for each class
                combined_labels = self.combine_labels_multi_thresholds(
                    winner_labels, thresholds=threshold, distances=distances
                )
            else:
                combined_labels = self.combine_labels(
                    winner_labels, threshold=threshold, distances=distances
                )  # Shape: (batch_size, H, W, C)

            end_time = time.time()
            if self.profile_time:
                print(
                    f"Time taken to combine labels: {end_time - start_time:.2f} seconds"
                )

            combined_labels = combined_labels.permute(
                0, 3, 1, 2
            ).long()  # Shape: (batch_size, C, H, W)

        return combined_labels

    def evaluate(
        self,
        k=10,
        distance_metric="cosine",
        weights="distance",
        threshold=0.5,
        config=None,
    ) -> dict:
        """
        Evaluate the KNN segmentation model on the validation set.

        Args:
            k (int): Number of nearest neighbors to use.
            distance_metric (str): Distance metric to use ("l2" or "cosine").
            weights (str): Weighting scheme to use ("distance" or "uniform").
            threshold (float or list of float): Threshold(s) for converting the averaged labels to binary.
            config (Namespace, optional): Configuration object containing parameters. If provided, will override instance parameters.
        Returns:
            dict: Dictionary containing the mean IoU for each class.
        """

        # Initialize representations and labels
        self.compute_representations(self.train_loader)

        classwise_ious = []

        with torch.no_grad():
            for batch in self.val_loader:

                # profile memory before processing batch
                if self.profile_memory:
                    print(
                        f"Current CUDA memory allocated before processing batch: {torch.cuda.memory_allocated() / 1024**2:.2f} MB"
                    )
                    # cpu memory
                    print(f"Current RAM usage: {psutil.virtual_memory().percent}%")
                class_mask = batch["class_mask"]
                class_mask = class_mask.permute(
                    0, 3, 1, 2
                ).long()  # Shape: (batch_size, C, H, W)

                x_test = batch["image"]
                combined_labels = self.predict(
                    x_test, k, distance_metric, weights, threshold
                )  # Shape: (batch_size, H, W, C)

                start_time = time.time()

                classwise_ious.append(self.iou_classwise(combined_labels, class_mask))
                print(f"Classwise IoUs: {classwise_ious[-1]}")
                end_time = time.time()
                if self.profile_time:
                    print(
                        f"Time taken to compute classwise IoUs: {end_time - start_time:.2f} seconds"
                    )

        # classwise_ious: [{"iou_class_1": iou_class_1, "iou_class_2": iou_class_2, ...}, ...]
        # Compute mean IoU for each class
        mean_ious = {}
        for c in classwise_ious[0]:
            # For each class
            iou_list = [iou[c] for iou in classwise_ious if iou[c] is not None]
            if iou_list:
                mean_ious[c] = np.mean(iou_list)
            else:
                mean_ious[c] = None

        print("Mean IoUs: ", mean_ious)
        return mean_ious
