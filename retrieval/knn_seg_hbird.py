# Adapted from: https://github.com/vpariza/open-hummingbird-eval/blob/main/hbird/hbird_eval.py

# Patch-level AOI segmentation based on the HummingBird approach. Added different aggregation strategies
# and the option of not averaging patch labels (default).

import torch
from data.retrieval_dataset import RetrievalDataset
import os
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
from pathlib import Path
from torchmetrics import JaccardIndex
from torchmetrics.wrappers import ClasswiseWrapper
from retrieval.hbird_eval_metrics import PredsmIoU
import albumentations as A
import cv2
import time
import faiss
from typing import Optional, List
from albumentations.core.transforms_interface import ImageOnlyTransform


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


class NearestNeighborSearchFaiss:
    def __init__(
        self,
        feature_memory,
        n_neighbors=30,
        distance_measure="dot_product",
        idx_shard=False,
        use_fp16=False,
        gpu_ids=None,
        **kwargs,
    ):
        self.n_neighbors = n_neighbors
        self.distance_measure = distance_measure.lower()
        self.idx_shard = idx_shard
        self.use_fp16 = use_fp16
        self.embed_d = feature_memory.size(1)

        self.n_gpus = faiss.get_num_gpus()  # Get available GPUs
        if self.n_gpus < 1:
            raise RuntimeError("No GPUs available for Faiss.")

        # Set GPU IDs to use
        if gpu_ids is None:
            gpu_ids = list(range(self.n_gpus))
        else:
            # Validate GPU IDs
            for gpu_id in gpu_ids:
                if gpu_id >= self.n_gpus or gpu_id < 0:
                    raise ValueError(
                        f"Invalid GPU ID: {gpu_id}. Available GPUs: 0-{self.n_gpus -1}"
                    )
        self.gpu_ids = gpu_ids

        # Initialize the Faiss index
        self.index = self._initialize_index()

        # Add feature vectors to the index
        self._add_features_to_index(feature_memory)

    def _get_sub_index(self, gpu_id, d: int):
        # Create resources for this GPU
        res = faiss.StandardGpuResources()

        # Create options for this GPU index
        config = faiss.GpuIndexFlatConfig()
        config.useFloat16 = self.use_fp16  # Use FP16 for better performance
        config.device = gpu_id  # Specify which GPU to use

        if self.distance_measure == "dot_product":
            return faiss.GpuIndexFlatIP(
                res, d, config
            )  # Inner Product (Dot Product) index
        elif self.distance_measure in ["l2", "euclidean"]:
            return faiss.GpuIndexFlatL2(res, d, config)  # L2 (Euclidean) distance index
        else:
            raise ValueError(f"Unsupported distance measure: {self.distance_measure}")

    def _initialize_index(self):
        d = self.embed_d

        if self.idx_shard:
            # Create shard index to distribute across GPUs
            gpu_index = faiss.IndexShards(d)
            gpu_index.threaded = True
            # Note: threaded=True means search will be done in parallel threads

            # For each GPU, create a GPU index and add it to the shards
            for i, gpu_id in enumerate(self.gpu_ids):
                gpu_sub_idx = self._get_sub_index(gpu_id, d)
                gpu_index.add_shard(gpu_sub_idx)

        else:
            gpu_index = faiss.IndexReplicas()

            # For each GPU, create a GPU index and add it to the replicas
            for i, gpu_id in enumerate(self.gpu_ids):
                # Create resources for this GPU
                gpu_sub_idx = self._get_sub_index(gpu_id, d)
                gpu_index.addIndex(gpu_sub_idx)

        return gpu_index

    def _add_features_to_index(self, feature_memory):
        # Ensure feature_memory is on the CPU before adding to the index
        feature_memory_cpu = feature_memory.cpu().numpy()
        self.index.add(feature_memory_cpu)

    def find_nearest_neighbors(self, q, k=None):
        if k is None:
            k = self.n_neighbors

        # Ensure query tensor is on the CPU
        q_cpu = q.cpu().numpy()
        distances, indices = self.index.search(q_cpu, k)
        return indices, distances


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


class KNNHummingBirdSegmentation:

    def __init__(
        self,
        encoder,
        classes=["wire", "ball", "wedge", "epoxy"],
        train_loader=None,
        val_loader=None,
        n_neighbours=30,
        num_batches=-1,
        config=None,
        num_sampled_features=1024,
        memory_size=None,
        pretrained_model=False,
        profile_time=False,
        profile_memory=False,
        normalize=False,
        average_patches=False,
        train_im_list=None,  # Overrides the train_loader if provided
        val_im_list=None,  # Overrides the val_loader if provided
        input_resolution=(512, 512),
        batch_size=32,
    ):

        self.encoder = encoder
        self.device = next(encoder.parameters()).device
        self.representations = None
        self.distances = None
        self.labels = None
        self.classes = classes
        self.num_classes = len(classes) if classes is not None else 0
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.profile_time = profile_time
        self.profile_memory = profile_memory
        self.batch_size = batch_size

        # HummingBird specific parameters
        self.num_batches = (
            num_batches  # batch size = 32, 31 batches, 10 features per image
        )
        self.num_sampled_features = None  # Number of patches to sample per image
        self.augmentation_epoch = 1  # Number of augmentation epochs
        self.n_neighbours = n_neighbours  # Number of nearest neighbors to consider
        self.nn_method = "faiss"  # Nearest neighbor search method
        self.threshold = None  # Threshold for uncertainty in segmentation
        self.beta = 0.02  # Temperature for cross-attention
        self.input_resolution = input_resolution  # Input resolution for the model
        self.patch_size = encoder.patch_embed.patch_size[0]

        if config is not None:
            self.classes = config.classes
            self.batch_size = config.batch_size
            self.data_path = Path(config.data_path)
            self.input_resolution = config.input_resolution
            self.num_classes = len(self.classes)
            normalize = config.normalize
            self.transform = self.create_transforms(normalize=normalize)

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

        # Memory configuration
        self.num_sampled_features = (
            1024 if num_sampled_features is None else num_sampled_features
        )  # Fixed number of features to sample per image
        self.total_batches = len(self.train_loader)
        self.max_batches = (
            config.max_batches if hasattr(config, "max_batches") else 1000
        )
        self.num_batches = min(
            self.num_batches, self.max_batches
        )  # Limit the number of batches to process
        self.memory_size = (
            self.batch_size * self.num_batches * self.num_sampled_features
        )  # batch size = 32, 31 batches, 10 features per image
        if self.num_batches == -1:
            self.memory_size = None  # Unbounded memory size
        if self.memory_size is not None:
            # create memory of specific size
            self.feature_memory = torch.zeros(
                (self.memory_size, self.encoder.embed_dim)
            )
            self.label_memory = torch.zeros((self.memory_size, self.num_classes))

        self.create_memory(self.train_loader, config, average_patches=average_patches)
        self.feature_memory = self.feature_memory.cpu()
        self.label_memory = self.label_memory.cpu()
        self.create_NN(self.n_neighbours, nn_method=self.nn_method)

    def create_loaders_from_list(self, im_list, batch_size=32, shuffle=False):

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

    def create_loaders(self, config, train_set_size=0.7):
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

    def create_NN(
        self, n_neighbours=30, nn_method="faiss", feature_memory=None, **kwargs
    ):

        feature_memory = (
            self.feature_memory if feature_memory is None else feature_memory
        )
        if self.nn_method == "faiss":
            self.NN_algorithm = NearestNeighborSearchFaiss(
                self.feature_memory, n_neighbors=n_neighbours, **kwargs
            )
        else:
            raise ValueError(
                f"Unsupported nearest neighbor method: {self.nn_method}. Supported methods: 'faiss'."
            )

    def find_nearest_key_to_query(
        self,
        q: torch.Tensor,
        k: Optional[int] = None,
        feature_memory: torch.Tensor = None,
        label_memory: torch.Tensor = None,
        return_distance: bool = False,
    ):
        """Find the nearest key features and labels to the query features.

        Args:
            q (torch.Tensor): Query tensor of shape (bs, num_patches, d_k).
            k (int, optional): Number of nearest neighbors to return. If None, uses self.n_neighbours.
            feature_memory (torch.Tensor, optional): Precomputed feature memory tensor of shape (num_features, d_k).
            label_memory (torch.Tensor, optional): Precomputed label memory tensor of shape (num_features, num_classes).
        Returns:
            key_features (torch.Tensor): Key features tensor of shape (bs, num_patches, k, d_k).
            key_labels (torch.Tensor): Key labels tensor of shape (bs, num_patches, k, num_classes).
        """

        if feature_memory is None:
            feature_memory = self.feature_memory
        if label_memory is None:
            label_memory = self.label_memory

        bs, num_patches, d_k = q.shape
        reshaped_q = q.reshape(bs * num_patches, d_k)
        # neighbors, distances = self.NN_algorithm.search_batched(reshaped_q)
        if k is None:
            k = self.n_neighbours

        num_pixels = (
            self.patch_size * self.patch_size
        )  # Assuming patch_size is defined in the class
        # distance shape -> (bs*num_patches, k)
        neighbors, distances = self.NN_algorithm.find_nearest_neighbors(reshaped_q, k=k)
        neighbors = neighbors.astype(np.int64)
        neighbors = torch.from_numpy(neighbors).cpu()
        neighbors = neighbors.flatten()
        key_features = self.feature_memory[neighbors]
        key_features = key_features.reshape(
            bs, num_patches, k, -1
        )  # shape (bs, num_patches, k, d_k)
        key_labels = self.label_memory[
            neighbors
        ]  # shape (bs*k, num_pixels, num_classes)
        key_labels = key_labels.reshape(
            bs, num_patches, k, num_pixels, -1
        )  # shape (bs, num_patches, k, patch_size*patch_size, num_classes)

        if return_distance:
            distances = torch.from_numpy(distances).float().cpu()
            distances = distances.reshape(bs, num_patches, k)
            return key_features, key_labels, distances
        else:
            return key_features, key_labels

    def get_patch_scores(self, gt, class_frequency):
        """Compute patch scores based on class frequency in each patch.

        Args:
            gt (torch.Tensor): Ground truth tensor of shape (bs, h, w, c) where c is the number of classes.
            class_frequency (torch.Tensor): Tensor of shape (num_classes,) containing the frequency of each class.
        """
        patch_scores = torch.zeros((gt.shape[0], gt.shape[1]))
        nonzero_indices = torch.zeros((gt.shape[0], gt.shape[1]), dtype=torch.bool)

        for i in range(gt.shape[0]):
            for j in range(gt.shape[1]):
                patch_classes = gt[i, j].unique()  # get unique classes in the patch
                patch_scores[i, j] = class_frequency[
                    patch_classes
                ].sum()  # sum of class frequencies in the patch
                nonzero_indices[i, j] = (
                    patch_classes.shape[0] > 0
                )  # if there are any classes in the patch

        return patch_scores, nonzero_indices

    def patchify_gt(self, gt, patch_size):
        bs, h, w, c = gt.shape

        # Reshape: (bs, h, w, c) -> (bs, h//patch_size, patch_size, w//patch_size, patch_size, c)
        gt = gt.reshape(bs, h // patch_size, patch_size, w // patch_size, patch_size, c)

        # Permute: (bs, h//patch_size, patch_size, w//patch_size, patch_size, c)
        #       -> (bs, h//patch_size, w//patch_size, patch_size, patch_size, c)
        gt = gt.permute(0, 1, 3, 2, 4, 5)

        # Final reshape: (bs, h//patch_size, w//patch_size, patch_size*patch_size, c)
        gt = gt.reshape(
            bs, h // patch_size, w // patch_size, patch_size * patch_size, c
        )

        return gt

    def cross_attention(self, q, k, v, beta=0.02):
        """
        Cross-attention with average label reconstruction.

        Args:
            q (torch.Tensor): query tensor of shape (bs, num_patches, d_k)
            k (torch.Tensor): key tensor of shape (bs, num_patches,  NN, d_k)
            v (torch.Tensor): value tensor of shape (bs, num_patches, NN, label_dim)

        """
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        q = q.unsqueeze(2)  # (bs, num_patches, 1, d_k)
        attn = torch.einsum("bnld,bnmd->bnlm", q, k) / beta  # (bs, num_patches, 1, NN)
        attn = attn.squeeze(2)  # remove previous unsqueeze
        attn = F.softmax(attn, dim=-1)
        attn = attn.unsqueeze(-1)
        label_hat = torch.einsum("blms,blmk->blsk", attn, v)  #
        label_hat = label_hat.squeeze(-2)
        return label_hat

    def cross_attention_simple_multi_beta(self, q, k, v, beta_per_class=None):
        """
        Simple approach - separate attention computation per class.
        """

        if beta_per_class is None:
            beta_per_class = [self.beta] * self.num_classes

        bs, num_patches, num_neighbors, num_classes = v.shape

        # Normalize features once
        q_norm = F.normalize(q, dim=-1)  # (bs, num_patches, d_k)
        k_norm = F.normalize(k, dim=-1)  # (bs, num_patches, NN, d_k)

        # Compute base attention scores
        base_scores = torch.einsum(
            "bpd,bpnd->bpn", q_norm, k_norm
        )  # (bs, num_patches, NN)

        class_predictions = []

        for class_idx, beta_class in enumerate(beta_per_class):
            # Apply class-specific temperature
            class_attn = F.softmax(
                base_scores / beta_class, dim=-1
            )  # (bs, num_patches, NN)

            # Get class-specific values
            class_values = v[:, :, :, class_idx]  # (bs, num_patches, NN)

            # Weighted sum for this class
            class_pred = torch.sum(
                class_attn * class_values, dim=-1, keepdim=True
            )  # (bs, num_patches, 1)
            class_predictions.append(class_pred)

        # Stack all class predictions
        label_hat = torch.cat(
            class_predictions, dim=-1
        )  # (bs, num_patches, num_classes)

        return label_hat

    def cross_attention_complete_patches(self, q, k, v, beta=0.02):
        """
        Cross-attention with complete patch label reconstruction.

        Args:
            q (torch.Tensor): query tensor of shape (bs, num_patches, d_k)
            k (torch.Tensor): key tensor of shape (bs, num_patches, NN, d_k)
            v (torch.Tensor): value tensor of shape (bs, num_patches, NN, patch_size*patch_size, num_classes)
        """
        # Normalize features for cosine similarity
        q = F.normalize(q, dim=-1)  # (bs, num_patches, d_k)
        k = F.normalize(k, dim=-1)  # (bs, num_patches, NN, d_k)

        # Compute attention weights
        q_expanded = q.unsqueeze(2)  # (bs, num_patches, 1, d_k)
        attn_scores = (
            torch.einsum("bnld,bnmd->bnlm", q_expanded, k) / beta
        )  # (bs, num_patches, 1, NN)
        attn_weights = F.softmax(
            attn_scores.squeeze(2), dim=-1
        )  # (bs, num_patches, NN)

        # Apply attention to complete patch labels
        attn_weights_expanded = attn_weights.unsqueeze(-1).unsqueeze(
            -1
        )  # (bs, num_patches, NN, 1, 1)

        # shape v: (bs, num_patches, NN, patch_size*patch_size, num_classes)
        # Weighted combination of complete patch labels
        reconstructed_patches = torch.sum(
            attn_weights_expanded * v, dim=2
        )  # (bs, num_patches, patch_size*patch_size, num_classes)

        return reconstructed_patches

    def sample_features_vectorized(self, features, patchified_gts):
        """Vectorized sample features based on the class frequency in the ground truth.

        Args:
            features (torch.Tensor): Features tensor of shape (bs, num_patches, d_k).
            patchified_gts (torch.Tensor): Ground truth tensor of shape (bs, spatial_resolution, spatial_resolution, c*patch_size*patch_size).
        """

        bs, _, _, _, _ = patchified_gts.shape

        # Vectorized class frequency computation
        class_frequency = self.get_class_frequency_vectorized(
            patchified_gts
        )  # Shape: (bs, num_classes)

        # Vectorized patch scores computation
        patch_scores, nonzero_indices = self.get_patch_scores_vectorized(
            patchified_gts, class_frequency
        )

        # Set invalid patches to high score (vectorized)
        patch_scores[~nonzero_indices] = 1e6

        # Generate uniform random values for all batches at once
        uniform_random = torch.rand_like(patch_scores)

        # Apply random scaling only to valid patches
        patch_scores[nonzero_indices] *= uniform_random[nonzero_indices]

        # Vectorized topk selection across all batches
        _, indices = torch.topk(
            patch_scores, self.num_sampled_features, largest=False, dim=1
        )

        # Vectorized feature sampling using advanced indexing
        batch_indices = (
            torch.arange(bs).unsqueeze(1).expand(-1, self.num_sampled_features)
        )
        sampled_features = features[
            batch_indices, indices
        ]  # Shape: (bs, num_sampled_features, d_k)

        return sampled_features, indices

    def get_class_frequency_vectorized(self, patchified_gts):
        """Vectorized version of class frequency computation."""
        # (bs, spatial_resolution, spatial_resolution, patch_size*patch_size, num_classes)
        bs, spatial_res_h, spatial_res_w, num_pixels, num_classes = patchified_gts.shape
        presence_indicators = (
            patchified_gts.sum(dim=3) > 0
        )  # Shape: (bs, spatial_res_h, spatial_res_w, num_classes)

        class_frequency = presence_indicators.sum(
            dim=(1, 2)
        )  # Shape: (bs, num_classes)

        return class_frequency

    def get_patch_scores_vectorized(self, patchified_gts, class_frequency):
        """Vectorized version of patch scores computation."""
        bs, spatial_res_h, spatial_res_w, num_pixels, num_classes = patchified_gts.shape

        presence_indicators = (
            patchified_gts.sum(dim=3) > 0
        )  # Shape: (bs, spatial_res_h, spatial_res_w, num_classes)
        # Reshape ground truth tensor to (bs, num_patches, num_classes)
        gt_reshaped = presence_indicators.view(
            bs, spatial_res_h * spatial_res_w, num_classes
        )  # Shape: (bs, num_patches, num_classes)
        # Expand class_frequency for broadcasting: (bs, 1, num_classes)
        class_freq_expanded = class_frequency.unsqueeze(1)

        # Compute patch scores: sum of class frequencies weighted by patch content
        # gt_reshaped: (bs, num_patches, num_classes)
        # class_freq_expanded: (bs, 1, num_classes)
        patch_scores = torch.sum(
            gt_reshaped * class_freq_expanded, dim=2
        ).float()  # Shape: (bs, num_patches)

        # Check if patches have any content (non-zero classes)
        nonzero_indices = torch.sum(gt_reshaped, dim=2) > 0  # Shape: (bs, num_patches)

        return patch_scores, nonzero_indices

    def sample_features(self, features, pathified_gts):
        """Sample features based on the class frequency in the ground truth.

        Args:
            features (torch.Tensor): Features tensor of shape (bs, num_patches, d_k).
            pathified_gts (torch.Tensor): Ground truth tensor of shape (bs, spatial_resolution, spatial_resolution, c*patch_size*patch_size).
        """
        sampled_features = []
        sampled_indices = []
        for k, gt in enumerate(pathified_gts):
            class_frequency = self.get_class_frequency(gt)
            patch_scores, nonzero_indices = self.get_patch_scores(gt, class_frequency)

            patch_scores = patch_scores.flatten()
            nonzero_indices = nonzero_indices.flatten()

            # assert zero_score_idx[0].size(0) != 0 # for pascal every patch should belong to one class
            patch_scores[~nonzero_indices] = 1e6

            # sample uniform distribution with the same size as the
            # number of nonzero indices (we use the sum here as the
            # nonzero_indices matrix is a boolean mask)
            uniform_x = torch.rand(nonzero_indices.sum())
            patch_scores[nonzero_indices] *= uniform_x
            feature = features[k]

            # select the least num_sampled_features score indices
            _, indices = torch.topk(
                patch_scores, self.num_sampled_features, largest=False
            )

            sampled_indices.append(indices)
            samples = feature[indices]
            sampled_features.append(samples)

        sampled_features = torch.stack(sampled_features)
        sampled_indices = torch.stack(sampled_indices)

        return sampled_features, sampled_indices

    def get_class_frequency(self, gt):
        class_frequency = torch.zeros((self.num_classes), device=self.device)
        # For each image in the batch
        for i in range(gt.shape[0]):
            # For each (row of) patch(es) in the image
            for j in range(gt.shape[1]):
                # Get unique classes in the patch
                patch_classes = gt[
                    i, j
                ].unique()  # preconfigured for multi-label segmentation
                class_frequency[patch_classes] += 1

        return class_frequency

    def create_memory(self, train_loader, config, average_patches=True):
        feature_memory = list()
        label_memory = list()
        idx = 0
        with torch.no_grad():
            for j in range(self.augmentation_epoch):
                time_start_access = time.perf_counter()
                for i, data in enumerate(train_loader):
                    if self.profile_time:
                        time_end_access = time.perf_counter()
                        print(
                            f"Time taken to access data: {time_end_access - time_start_access:.2f} seconds"
                        )
                        time_start_access = time_end_access
                    if self.profile_memory:
                        print(
                            f"Memory usage cuda: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB"
                        )
                    x = data["image"].to(self.device)
                    y = data["class_mask"].to(self.device)
                    # Rescaling probably unnecessary
                    y = y.long()

                    features = self.encoder(x)
                    features = features[
                        "x_norm_patchtokens"
                    ]  # Shape: (bs, num_patches, d_k)
                    normalized_features = features / torch.norm(
                        features, dim=2, keepdim=True
                    )
                    assert torch.allclose(
                        torch.norm(normalized_features, dim=2), torch.tensor(1.0)
                    ), "Features should be normalized"

                    patch_size = self.encoder.patch_embed.patch_size[0]
                    patchified_gts = self.patchify_gt(
                        y, patch_size
                    )  # (bs, spatial_resolution, spatial_resolution, patch_size*patch_size, num_classes)
                    # Compute the label for each patch
                    if average_patches:
                        label = patchified_gts.float().mean(dim=3)
                    else:
                        label = patchified_gts.float()

                    if self.memory_size is None:
                        # Memory Size is unbounded so we store all the features
                        # Spatially flattened feature map
                        normalized_features = normalized_features.flatten(
                            0, 1
                        )  # shape: (bs * num_patches, d_k)
                        label = label.flatten(
                            0, 2
                        )  # flatten the label map to match the features
                        # shape: (bs * num_patches, num_classes)
                        feature_memory.append(normalized_features.detach().cpu())
                        label_memory.append(label.detach().cpu())
                    else:
                        # Sample a fixed number of features per image to fit in limited size memory
                        # TODO: Test vectorized sampling, hasn't been needed yet with current number of images
                        # and image sizes.

                        sampled_features, sampled_indices = (
                            self.sample_features_vectorized(features, patchified_gts)
                        )
                        normalized_sampled_features = sampled_features
                        label = label.flatten(1, 2)
                        # select the labels of the sampled features
                        sampled_indices = sampled_indices.to(self.device)

                        # repeat the label for each sampled feature
                        label_hat = label.gather(
                            1,
                            sampled_indices.unsqueeze(-1).expand(
                                -1, -1, label.shape[-1]
                            ),
                        )

                        # label_hat = label.gather(1, sampled_indices)
                        normalized_sampled_features = (
                            normalized_sampled_features.flatten(0, 1)
                        )
                        label_hat = label_hat.flatten(0, 1)
                        self.feature_memory[
                            idx : (idx + normalized_sampled_features.size(0))
                        ] = normalized_sampled_features.detach().cpu()
                        self.label_memory[idx : idx + label_hat.size(0)] = (
                            label_hat.detach().cpu()
                        )
                        idx += normalized_sampled_features.size(0)
                        # memory.append(normalized_sampled_features.detach().cpu())
                        if idx + 1 >= self.memory_size:
                            break  # Stop when memory is full
                    if self.profile_time:
                        time_start_access = time.perf_counter()

            if self.memory_size is None:
                self.feature_memory = torch.cat(feature_memory)
                self.label_memory = torch.cat(label_memory)

    def multi_scale_cross_attention(self, q, k, v, beta=0.02):
        """
        Multi-scale cross-attention that combines patch-level and pixel-level information.

        In multi-scale attention, we first compute patch-level predictions using the average label of each patch.
        Then, we compute pixel-level predictions within each patch using the full patch labels.
        Finally, we combine these two predictions to get a more refined output.

        Args:
            q: query features (bs, num_patches, d_k)
            k: key features (bs, num_patches, NN, d_k)
            v: complete patch labels (bs, num_patches, NN, patch_size*patch_size*num_classes)
        """

        # TODO: test multi-scale attention thoroughly.

        patch_size = int(np.sqrt(v.shape[-1] // self.num_classes))

        # 1. Patch-level attention (your current approach)
        v_patch_avg = v.view(
            *v.shape[:-1], patch_size * patch_size, self.num_classes
        ).mean(
            dim=-2
        )  # Average within patches
        patch_predictions = self.cross_attention(
            q, k, v_patch_avg, beta
        )  # (bs, num_patches, num_classes)

        # 2. Pixel-level attention within patches
        v_reshaped = v.view(
            *v.shape[:-1], patch_size * patch_size, self.num_classes
        )  # (bs, num_patches, NN, pixels_per_patch, num_classes)

        # Compute attention weights (same as before)
        q_norm = F.normalize(q, dim=-1)
        k_norm = F.normalize(k, dim=-1)
        attn_weights = F.softmax(
            torch.einsum("bpd,bpnd->bpn", q_norm, k_norm) / beta, dim=-1
        )  # (bs, num_patches, NN)

        # Apply attention to each pixel within patches
        attn_weights_expanded = attn_weights.unsqueeze(-1).unsqueeze(
            -1
        )  # (bs, num_patches, NN, 1, 1)
        pixel_predictions = torch.sum(
            attn_weights_expanded * v_reshaped, dim=2
        )  # (bs, num_patches, pixels_per_patch, num_classes)

        # 3. Combine patch and pixel predictions
        patch_pred_expanded = patch_predictions.unsqueeze(-2).expand(
            -1, -1, patch_size * patch_size, -1
        )  # (bs, num_patches, pixels_per_patch, num_classes)

        # Weighted combination
        alpha = 0.3  # Weight for patch-level predictions
        combined_predictions = (
            alpha * patch_pred_expanded + (1 - alpha) * pixel_predictions
        )

        return combined_predictions.view(
            q.shape[0], q.shape[1], -1
        )  # Flatten back to (bs, num_patches, patch_size*patch_size*num_classes)

    def reshape_patches_to_spatial(self, predictions, original_h, original_w):
        """
        Reshape predictions from patch format to spatial format.

        Args:
            predictions: (B, num_patches, pixels_per_patch, num_classes)
            original_h, original_w: Target spatial dimensions

        Returns:
            reshaped: (B, num_classes, original_h, original_w)
        """
        B, num_patches, pixels_per_patch, num_classes = predictions.shape

        # Calculate spatial resolution at patch level
        spatial_resolution = int(np.sqrt(num_patches))  # Assumes square patch grid

        # Step 1: Reshape pixels_per_patch back to patch spatial dimensions
        predictions = predictions.view(
            B,
            spatial_resolution,
            spatial_resolution,  # patch grid
            self.patch_size,
            self.patch_size,  # pixels within each patch
            num_classes,
        )  # (B, spatial_res, spatial_res, patch_size, patch_size, num_classes)

        # Step 2: Rearrange to reconstruct full spatial image
        predictions = predictions.permute(
            0, 5, 1, 3, 2, 4
        )  # (B, num_classes, spatial_res, patch_size, spatial_res, patch_size)

        # Step 3: Merge patch dimensions with spatial dimensions
        patch_h = spatial_resolution * self.patch_size
        patch_w = spatial_resolution * self.patch_size
        predictions = predictions.contiguous().view(
            B, num_classes, patch_h, patch_w
        )  # (B, num_classes, H_patches, W_patches)

        # Step 4: Interpolate to original resolution if needed
        if patch_h != original_h or patch_w != original_w:
            predictions = F.interpolate(
                predictions,
                size=(original_h, original_w),
                mode="bilinear",
                align_corners=False,
            )

        return predictions

    def predict_average(
        self,
        x_test: torch.Tensor,
        k: Optional[int] = None,
        feature_memory: Optional[torch.Tensor] = None,
        label_memory: Optional[torch.Tensor] = None,
        beta: Optional[int] = None,
    ) -> torch.Tensor:
        """Predict using cross-attention with average label reconstruction.

        Args:
            x_test (torch.Tensor): Input tensor of shape (bs, c, h, w).
            k (int, optional): Number of nearest neighbors to consider. If None, uses self.n_neighbours.
            feature_memory (torch.Tensor, optional): Precomputed feature memory tensor of shape (num_features, d_k).
            label_memory (torch.Tensor, optional): Precomputed label memory tensor of shape (num_features, num_classes).
            beta (float, optional): Temperature parameter for attention. If None, uses self.beta.
        Returns:
            torch.Tensor: Predicted labels of shape (bs, num_classes, h, w).
        """

        self.encoder.eval()

        if beta is None:
            beta = self.beta

        eval_spatial_resolution = self.input_resolution[0] // self.encoder.patch_size
        with torch.no_grad():

            x_test = x_test.to(self.device)

            h = x_test.shape[2]
            w = x_test.shape[3]

            features = self.encoder(x_test)[
                "x_norm_patchtokens"
            ]  # Shape: (batch_size, num_patches, d_k)
            features_normalized = features / torch.norm(features, dim=2, keepdim=True)
            features_normalized = features_normalized.to("cpu")
            q = features_normalized.clone().detach()
            key_features, key_labels = self.find_nearest_key_to_query(
                q, feature_memory=feature_memory, label_memory=label_memory, k=k
            )
            label_hat = self.cross_attention(
                features_normalized, key_features, key_labels, beta
            )
            bs, _, label_dim = label_hat.shape
            combined_labels = label_hat.reshape(
                bs, eval_spatial_resolution, eval_spatial_resolution, label_dim
            ).permute(0, 3, 1, 2)
            resized_label_hats = F.interpolate(
                combined_labels.float(), size=(h, w), mode="bilinear"
            )

        return resized_label_hats

    def predict_multi_betas(
        self,
        x_test: torch.Tensor,
        k: Optional[int] = None,
        feature_memory: Optional[torch.Tensor] = None,
        label_memory: Optional[torch.Tensor] = None,
        beta_per_class: Optional[List[float]] = None,
    ) -> torch.Tensor:
        """Predict using multi-scale cross-attention with separate betas for each class.
        Args:
            x_test (torch.Tensor): Input tensor of shape (bs, c, h, w
            k (int, optional): Number of nearest neighbors to consider. If None, uses self.n_neighbours.
            feature_memory (torch.Tensor, optional): Precomputed feature memory tensor of shape (num_features
            label_memory (torch.Tensor, optional): Precomputed label memory tensor of shape (num_features, num_classes).
            beta_per_class (List[float], optional): List of betas for each class. If None, uses self.beta for all classes.
        Returns:
            torch.Tensor: Predicted labels of shape (bs, num_classes, h, w).
        """
        self.encoder.eval()

        eval_spatial_resolution = self.input_resolution[0] // self.encoder.patch_size
        with torch.no_grad():

            x_test = x_test.to(self.device)

            h = x_test.shape[2]
            w = x_test.shape[3]

            features = self.encoder(x_test)[
                "x_norm_patchtokens"
            ]  # Shape: (batch_size, num_patches, d_k)
            features_normalized = features / torch.norm(features, dim=2, keepdim=True)
            features_normalized = features_normalized.to("cpu")
            q = features_normalized.clone().detach()
            key_features, key_labels = self.find_nearest_key_to_query(
                q, feature_memory=feature_memory, label_memory=label_memory, k=k
            )
            label_hat = self.cross_attention_simple_multi_beta(
                features_normalized, key_features, key_labels, beta_per_class
            )
            bs, _, label_dim = label_hat.shape
            combined_labels = label_hat.reshape(
                bs, eval_spatial_resolution, eval_spatial_resolution, label_dim
            ).permute(0, 3, 1, 2)
            resized_label_hats = F.interpolate(
                combined_labels.float(), size=(h, w), mode="bilinear"
            )

        return resized_label_hats

    def predict_complete_patches(
        self,
        x_test: torch.Tensor,
        k: Optional[int] = None,
        feature_memory: Optional[torch.Tensor] = None,
        label_memory: Optional[torch.Tensor] = None,
        beta: Optional[float] = None,
        attention_mechanism="normal",
    ):
        """Predict using cross-attention with complete patch reconstruction.

        Args:
            x_test (torch.Tensor): Input tensor of shape (bs, c, h, w).
            k (int, optional): Number of nearest neighbors to consider. If None, uses self.n_neighbours.
            feature_memory (torch.Tensor, optional): Precomputed feature memory tensor of shape (num_features, d_k).
            label_memory (torch.Tensor, optional): Precomputed label memory tensor of shape (num_features, num_classes).
            beta (float, optional): Temperature parameter for attention. If None, uses self.beta.
            attention_mechanism (str): Type of attention mechanism to use ("normal" or "multi_scale").
        Returns:
            torch.Tensor: Predicted labels of shape (bs, num_classes, h, w).
        """

        self.encoder.eval()

        if beta is None:
            beta = self.beta

        with torch.no_grad():
            x_test = x_test.to(self.device)
            h, w = x_test.shape[2], x_test.shape[3]

            # Get features
            features = self.encoder(x_test)[
                "x_norm_patchtokens"
            ]  # (bs, num_patches, d_k)
            features_normalized = F.normalize(features, dim=-1)
            features_normalized = features_normalized.to("cpu")

            # Find nearest neighbors
            # key_features shape: (bs, num_patches, k, d_k)
            # key_labels shape: (bs, num_patches, k, patch_size*patch
            key_features, key_labels = self.find_nearest_key_to_query(
                features_normalized,
                k=k,
                feature_memory=feature_memory,
                label_memory=label_memory,
            )

            # Cross-attention with complete patches
            if attention_mechanism == "multi_scale":
                reconstructed_patches = self.multi_scale_cross_attention(
                    features_normalized, key_features, key_labels, beta
                )
            else:
                reconstructed_patches = self.cross_attention_complete_patches(
                    features_normalized, key_features, key_labels, beta
                )  # (bs, num_patches, patch_size*patch_size, num_classes)

            # Reshape to spatial format
            reshaped_patches = self.reshape_patches_to_spatial(
                reconstructed_patches, h, w
            )

        return reshaped_patches

    def _combine_labels_distance(self, label_masks, threshold=0.5, distances=None):
        # label_masks: (B, num_patches,  ,num_neighbors, C)
        # distances: (B, H*W, K) or (B, num_patches, K)
        # [32, 1024, 5, 4]

        if distances is not None:
            # Reshape distances to match label_masks
            # distances: (B*H*W, K) -> (B, H, W, K, 1)
            B, num_patches, K = distances.shape

            # Compute weights: inverse of distance (closer patches get higher weight)
            weights = 1 / (distances + 1e-8)
            weights = weights / weights.sum(dim=2, keepdim=True)  # Normalize over K
            # Weighted sum over K nearest patches

            # Expand weights to match label_masks dimensions
            weights_expanded = weights.unsqueeze(-1).unsqueeze(
                -1
            )  # (B, num_patches, K, 1, 1)

            # Weighted combination
            weighted_labels = label_masks * weights_expanded
            mean_labels = weighted_labels.sum(
                dim=2
            )  # (B, num_patches, num_pixels, num_classes)
        else:
            # Simple mean over K neighbors
            mean_labels = label_masks.mean(dim=2)  # (B, num_patches, C)

        return mean_labels  # B, num_patches, num_pixels, C

    def predict_distance(
        self,
        x_test: torch.Tensor,
        k: Optional[int] = None,
        feature_memory: Optional[torch.Tensor] = None,
        label_memory: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict distances using the nearest neighbors in the feature memory.
        Args:
            x_test (torch.Tensor): Input tensor of shape (bs, c, h, w).
            k (int, optional): Number of nearest neighbors to consider. If None, uses self.n_neighbours.
            feature_memory (torch.Tensor, optional): Precomputed feature memory tensor of shape (num_features, d_k).
            label_memory (torch.Tensor, optional): Precomputed label memory tensor of shape (num_features, num_classes).
        Returns:
            torch.Tensor: Predicted labels of shape (bs, num_classes, h, w).
        """

        self.encoder.eval()

        with torch.no_grad():

            x_test = x_test.to(self.device)

            h = x_test.shape[2]
            w = x_test.shape[3]

            features = self.encoder(x_test)[
                "x_norm_patchtokens"
            ]  # Shape: (batch_size, num_patches, d_k)
            features_normalized = features / torch.norm(features, dim=2, keepdim=True)
            features_normalized = features_normalized.to("cpu")
            q = features_normalized.clone().detach()
            key_features, key_labels, similarities = self.find_nearest_key_to_query(
                q,
                feature_memory=feature_memory,
                label_memory=label_memory,
                k=k,
                return_distance=True,
            )

            # print(f"Average similarities: {similarities.mean()}")  # Debugging line to check distances

            label_hat = self._combine_labels_distance(
                key_labels, distances=(1 - similarities)
            )  # (bs, num_patches, num_classes)
            predictions = self.reshape_patches_to_spatial(
                label_hat, h, w
            )  # (bs, num_classes, h, w)

        return predictions

    def evaluate_distance(self, k=10, threshold=0.5) -> dict:
        """
        Evaluation loop using distance-based prediction.

        Args:
            k (int): Number of nearest neighbors to consider.
            threshold (float): Threshold for converting probabilities to binary predictions.
        Returns:
            dict: Mean IoU for each class.
        """

        classwise_ious = []

        with torch.no_grad():
            for batch in self.val_loader:

                class_mask = (
                    batch["class_mask"].permute(0, 3, 1, 2).long()
                )  # Change shape to (B, C, H, W)

                x_test = batch["image"]
                combined_labels = self.predict_distance(
                    x_test,
                    k,
                    feature_memory=self.feature_memory,
                    label_memory=self.label_memory,
                )  # Shape: (batch_size, H, W, C)

                combined_labels_thresholded = (
                    combined_labels > threshold
                ).float()  # Thresholding the predictions

                start_time = time.time()

                classwise_ious.append(
                    self.iou_classwise(combined_labels_thresholded, class_mask)
                )
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

    def evaluate(
        self,
        patch_mechanism="complete",
        attention_mechanism="normal",
        k=None,
        thresholds=None,
        betas=None,
    ):
        """
        Evaluation loop of the HummingBird classifier using attention.

        Args:
            patch_mechanism (str): "average" or "complete" for patch label aggregation
            attention_mechanism (str): "normal" or "multi_scale" for attention type
            k (int, optional): Number of nearest neighbors to consider. If None, uses self.n_neighbours.
            thresholds (list, optional): List of thresholds for IoU computation. If None, uses default.
            betas (float or list, optional): Temperature parameter(s) for attention. If list, should match number of classes.
        Returns:
            dict: Mean IoU for each class.
        """

        feature_memory = self.feature_memory
        label_memory = self.label_memory

        # PredsmIoU applies matching because hummingbird is a clustering method
        if thresholds is None:
            metric = PredsmIoU(self.num_classes, self.num_classes)
        else:
            metric = PredsmIoU(self.num_classes, self.num_classes, threshold=thresholds)

        labels = []
        labels_hat = []
        idx = 0
        mean_ious = {k: [] for k in self.classes}

        with torch.no_grad():
            for batch in self.val_loader:
                labels = []
                labels_hat = []
                # PredsmIoU applies matching because hummingbird is a clustering method
                # Turns out mathing is not necessary in the end because patch masks are not averaged.
                if thresholds is None:
                    metric = PredsmIoU(self.num_classes, self.num_classes)
                else:
                    metric = PredsmIoU(
                        self.num_classes, self.num_classes, threshold=thresholds
                    )

                class_mask = batch["class_mask"]
                class_mask = class_mask
                labels.append(class_mask.detach())
                x_test = batch["image"]
                if patch_mechanism == "average":
                    if isinstance(betas, list):
                        resized_label_hats = self.predict_multi_betas(
                            x_test,
                            k=k,
                            beta_per_class=betas,
                            feature_memory=feature_memory,
                            label_memory=label_memory,
                        ).permute(
                            0, 2, 3, 1
                        )  # Shape: (batch_size, H, W, C)
                    else:
                        resized_label_hats = self.predict_average(
                            x_test,
                            k=k,
                            beta=betas,
                            feature_memory=feature_memory,
                            label_memory=label_memory,
                        ).permute(
                            0, 2, 3, 1
                        )  # Shape: (batch_size, H, W, C)
                elif patch_mechanism == "complete":
                    resized_label_hats = self.predict_complete_patches(
                        x_test,
                        k=k,
                        beta=betas,
                        feature_memory=feature_memory,
                        label_memory=label_memory,
                        attention_mechanism=attention_mechanism,
                    )
                    resized_label_hats = resized_label_hats.permute(
                        0, 2, 3, 1
                    )  # Shape: (batch_size, H, W, C)

                labels_hat.append(resized_label_hats.detach())

                idx += 1

                labels = torch.cat(labels)
                labels_hat = torch.cat(labels_hat)
                metric.update(labels, labels_hat)
                jac, tp, fp, fn = metric.compute(
                    is_global_zero=True, thresholds=thresholds
                )
                print(
                    f"True Positives: {tp}, False Positives: {fp}, False Negatives: {fn}"
                )
                print("Mean IoUs: ", jac)

                # Collect mean IoUs for each class
                m_ious_keylist = list(mean_ious.keys())
                for i, m_iou in enumerate(jac):
                    # Skip if m_iou is None
                    if m_iou is not None:
                        curr_key = m_ious_keylist[i]
                        mean_ious[curr_key].append(m_iou)

        mean_ious = {k: np.mean(v) if v else None for k, v in mean_ious.items()}
        return mean_ious

    def test(
        self,
        config,
        patch_mechanism="average",
        attention_mechanism="normal",
        k=None,
        thresholds=None,
        beta=None,
    ):
        """
        Evaluation loop of the HummingBird classifier.
        """

        # TODO:

        feature_memory = self.feature_memory
        label_memory = self.label_memory

        # PredsmIoU applies matching because hummingbird is a clustering method
        if thresholds is None:
            metric = PredsmIoU(self.num_classes, self.num_classes)
        else:
            metric = PredsmIoU(self.num_classes, self.num_classes, threshold=thresholds)

        labels = []
        labels_hat = []
        idx = 0
        with torch.no_grad():
            for batch in self.test_loader:

                class_mask = batch["class_mask"]
                class_mask = class_mask
                labels.append(class_mask.detach())
                x_test = batch["image"]
                if patch_mechanism == "average":
                    if isinstance(beta, list):
                        resized_label_hats = self.predict_multi_betas(
                            x_test,
                            k=k,
                            beta_per_class=beta,
                            feature_memory=feature_memory,
                            label_memory=label_memory,
                        ).permute(
                            0, 2, 3, 1
                        )  # Shape: (batch_size, H, W, C)
                    else:
                        resized_label_hats = self.predict_average(
                            x_test,
                            k=k,
                            beta=beta,
                            feature_memory=feature_memory,
                            label_memory=label_memory,
                        ).permute(
                            0, 2, 3, 1
                        )  # Shape: (batch_size, H, W, C)
                elif patch_mechanism == "complete":
                    resized_label_hats = self.predict_complete_patches(
                        x_test,
                        k=k,
                        beta=beta,
                        feature_memory=feature_memory,
                        label_memory=label_memory,
                        attention_mechanism=attention_mechanism,
                    ).permute(
                        0, 2, 3, 1
                    )  # Shape: (batch_size, H, W, C)

                labels_hat.append(resized_label_hats.detach())

                idx += 1

        labels = torch.cat(labels)
        labels_hat = torch.cat(labels_hat)
        metric.update(labels, labels_hat)
        jac, tp, fp, fn = metric.compute(is_global_zero=True, thresholds=thresholds)
        print(f"True Positives: {tp}, False Positives: {fp}, False Negatives: {fn}")
        print("Mean IoUs: ", jac)
        return jac
