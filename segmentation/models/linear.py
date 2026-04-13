import torch
from data.retrieval_dataset import RetrievalDataset
import os
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
from torchmetrics import JaccardIndex
from torchmetrics.wrappers import ClasswiseWrapper
import albumentations as A
import cv2
import time
from tqdm.notebook import tqdm
from torch import nn
from torch.nn.functional import interpolate
from lightning.pytorch import Callback
from data.retrieval_module import RetrievalDataModule
from torch.utils.data import TensorDataset
from segmentation_models_pytorch.losses import DiceLoss
import torch.nn.functional as F


class LinearClassifier(nn.Module):
    def __init__(self, dim, num_labels=4):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Conv2d(
            in_channels=dim,  # Total feature channels after fusion
            out_channels=num_labels,  # Number of segmentation classes
            kernel_size=1,  # 1x1 conv for pixel-wise classification
            stride=1,
            padding=0,
        )
        self.bn = nn.BatchNorm2d(dim)

    def forward(self, x):
        # x: [B, Num_patches, Embed_dim]
        x = self.bn(x)
        x = self.linear(x)
        return x


class LinearPatchClassifier:
    """Per-patch logistic regression for semantic segmentation"""

    def __init__(
        self,
        encoder,
        classes=None,
        num_epochs=100,
        lr=1e-3,
        train_loader=None,
        val_loader=None,
        debug=False,
        config=None,
        depth=2,
        threshold=0.5,
    ):

        self.encoder = encoder.eval()
        self.patch_size = self.encoder.patch_embed.patch_size[0]
        self.device = next(encoder.parameters()).device
        self.classes = classes
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss = self.focal_loss
        self.embedding_dimension = self.encoder.embed_dim
        self.debug = debug
        self.profile_memory = debug
        self.lr = lr
        self.num_epochs = num_epochs
        self.depth = self.encoder.n_blocks if depth is None else depth
        self.channels = (
            self.embedding_dimension * self.depth
        )  # Total channels after concatenation
        self.threshold = threshold  # Threshold for binary classification

        if config is not None:
            self.classes = config.classes
            self.batch_size = config.batch_size
            self.data_path = Path(config.data_path)
            self.input_resolution = config.input_resolution
            self.profile_time = config.profile_time
            normalization = A.Normalize(
                mean=[0.1872, 0.2352],  # Pretrain dataset mean
                std=[0.1924, 0.25308],  # Pretrain dataset std
            )

            # Transformations for training and validation
            self.transform = [
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
            if config.normalize:
                self.transform.append(normalization)
            self.transform = A.Compose(self.transform)

            # Transformations for random cropping evaluations
            self.transform2 = A.Compose(
                [
                    A.CropNonEmptyMaskIfExists(
                        self.input_resolution[0], self.input_resolution[1]
                    )
                ]
            )

            self.create_loaders(config)

        self.iou_classwise = ClasswiseWrapper(
            JaccardIndex(
                task="multilabel",
                num_labels=len(self.classes),
                average="none",
            ),
            labels=self.classes,
            prefix="iou_",
        ).to(self.device)
        self.num_classes = len(self.classes)
        self.classifier = LinearClassifier(
            dim=self.channels, num_labels=self.num_classes
        ).to(self.device)
        print(
            f"Linear classifier initialized with {self.embedding_dimension} embedding dimension and {self.num_classes} classes."
        )
        self.create_memory(self.all_loader, config)

    def create_loaders(self, config):
        im_list = [
            self.data_path / "train" / img_file
            for img_file in os.listdir(self.data_path / "train")
        ]

        self.train_indices = np.random.choice(
            len(im_list), int(0.8 * len(im_list)), replace=False
        )

        self.val_indices = np.array(
            list(set(range(len(im_list))) - set(self.train_indices))
        )

        im_list = np.array(im_list)
        np.random.shuffle(im_list)
        train_im_list = im_list[self.train_indices].tolist()
        val_im_list = im_list[self.val_indices].tolist()

        # Train and validation loaders
        self.aoi_train = RetrievalDataset(
            train_im_list,
            self.classes,
            transform=self.transform,
        )

        self.aoi_val = RetrievalDataset(
            val_im_list,
            self.classes,
            transform=self.transform,
        )

        self.aoi_all = RetrievalDataset(
            im_list,
            self.classes,
            transform=self.transform,
        )

        self.train_loader = DataLoader(
            self.aoi_train,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4,
        )

        self.val_loader = DataLoader(
            self.aoi_val,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=4,
        )

        self.all_loader = DataLoader(
            self.aoi_all,
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

    def extract_and_upsample_features(self, x, config):
        """Extract multi-layer features and upsample to input resolution"""

        # Get intermediate layer features - returns list of (B, num_patches, embed_dim)
        features = self.encoder.get_intermediate_layers(x, self.depth)

        # Convert patch features to spatial format and upsample
        spatial_features = []

        for feature in features:
            # feature shape: (batch, num_patches, embed_dim)
            batch_size, num_patches, embed_dim = feature.shape

            # Calculate spatial dimensions
            patches_per_side = int(num_patches**0.5)
            assert (
                patches_per_side * patches_per_side == num_patches
            ), f"num_patches {num_patches} is not a perfect square"

            # Reshape to spatial format: (batch, embed_dim, height, width)
            spatial_feature = feature.permute(
                0, 2, 1
            )  # (batch, embed_dim, num_patches)
            spatial_feature = spatial_feature.view(
                batch_size, embed_dim, patches_per_side, patches_per_side
            )  # (batch, embed_dim, patch_h, patch_w)

            spatial_features.append(spatial_feature)

        spatial_features = torch.concatenate(
            spatial_features, dim=1
        )  # Concatenate along the channel dimension

        return spatial_features

    def create_memory(self, all_loader: torch.utils.data.DataLoader, config):
        feature_memory = list()
        label_memory = list()
        ignore_mask_memory = list()
        print(f"Creating memory for {len(all_loader.dataset)} images...")
        with torch.no_grad():
            time_start_access = time.perf_counter()
            for i, data in enumerate(tqdm(all_loader, desc="Memory Creation loop")):
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
                y = data[
                    "class_mask"
                ].detach()  # Shape: (bs, num_patches, pixels_per_patch, num_classes)
                ignore_mask = data[
                    "ignore_mask"
                ].detach()  # Shape: (bs, num_patches, pixels_per_patch, num_classes)

                features = self.extract_and_upsample_features(
                    x, config
                )  # List of features for each layer

                # Compute the label for each patch

                feature_memory.append(features.detach().cpu())
                label_memory.append(y.detach().cpu())
                ignore_mask_memory.append(ignore_mask.detach().cpu())

                if self.profile_time:
                    time_start_access = time.perf_counter()

            print(f"Total patches processed: {len(feature_memory)}")
            print(f"Total labels processed: {len(label_memory)}")
            print(
                "Patch feature size: " + str(feature_memory[0].shape)
                if feature_memory
                else "No features processed"
            )
            print(
                "Label size: " + str(label_memory[0].shape)
                if label_memory
                else "No labels processed"
            )
            feature_memory = torch.cat(
                feature_memory, dim=0
            )  # Shape: (b, num_patches, d_k)
            label_memory = torch.cat(
                label_memory, dim=0
            )  # Shape: (b, num_patches, num_classes)
            ignore_mask_memory = torch.cat(
                ignore_mask_memory, dim=0
            )  # Shape: (b, num_patches, num_classes)
            print(f"Final feature memory shape: {feature_memory.shape}")
            print(f"Final label memory shape: {label_memory.shape}")
            print(f"Final ignore mask memory shape: {ignore_mask_memory.shape}")
        # Create TensorDataset and DataLoader
        tensor_train_indices = torch.tensor(self.train_indices)
        tensor_val_indices = torch.tensor(self.val_indices)
        feature_memory_flattened_train = feature_memory[
            tensor_train_indices
        ]  # b*num_patches, d_k
        label_memory_flattened_train = label_memory[
            tensor_train_indices
        ]  # b*num_patches, pixels_per_patch, num_classes
        ignore_mask_memory_flattened_train = ignore_mask_memory[
            tensor_train_indices
        ]  # b*num_patches, pixels_per_patch, num_classes

        feature_memory_flattened_val = feature_memory[
            tensor_val_indices
        ]  # b*num_patches, d_k
        label_memory_flattened_val = label_memory[
            tensor_val_indices
        ]  # b*num_patches, pixels_per_patch, num_classes
        ignore_mask_memory_flattened_val = ignore_mask_memory[
            tensor_val_indices
        ]  # b*num_patches, pixels_per_patch, num_classes

        tensor_dataset_train = TensorDataset(
            feature_memory_flattened_train,
            label_memory_flattened_train,
            ignore_mask_memory_flattened_train,
        )
        tensor_dataset_val = TensorDataset(
            feature_memory_flattened_val,
            label_memory_flattened_val,
            ignore_mask_memory_flattened_val,
        )

        # Create DataLoader with desired batch size
        patch_dataloader_train = DataLoader(
            tensor_dataset_train,
            batch_size=256,  # Adjust batch size for patch training
            shuffle=True,
            num_workers=0,  # Use 0 since data is already in memory
            pin_memory=True,  # Faster GPU transfer
        )

        # Create DataLoader with desired batch size
        patch_dataloader_val = DataLoader(
            tensor_dataset_val,
            batch_size=256,  # Adjust batch size for patch training
            shuffle=True,
            num_workers=0,  # Use 0 since data is already in memory
            pin_memory=True,  # Faster GPU transfer
        )

        self.patch_dataloader_train = patch_dataloader_train
        self.patch_dataloader_val = patch_dataloader_val

    def patchify_gt(self, gt, patch_size):
        bs, h, w, c = gt.shape
        gt = gt.reshape(bs, c, h // patch_size, patch_size, w // patch_size, patch_size)
        gt = gt.permute(0, 2, 4, 1, 3, 5)
        # bs, h//patch_size, w//patch_size, patch_size*patch_size, c
        gt = gt.reshape(
            bs, h // patch_size, w // patch_size, patch_size * patch_size, c
        )
        gt = gt.flatten(
            1, 2
        )  # Flatten the patch dimensions -> bs, num_patches, patch_size*patch_size, c
        gt = gt.mean(dim=-2)  # Average over the patch pixels -> bs, num_patches, c
        return gt  # bs, h//patch_size, w//patch_size, patch_size*patch_size, c

    def extract_features(self, x):
        """Extracts features from the input image using the encoder"""
        x = x.to(self.device)
        features = self.encoder(x)
        features = features["x_norm_patchtokens"]
        return features

    def predict(self, features: torch.Tensor, h, w):
        """Predicts the class for each patch in the input image"""
        p = self.patch_size
        y_hat = self.classifier(features)  # [B, Num_patches, C]
        y_hat = y_hat.permute(0, 2, 1)  # [B, C, Num_patches]
        y_hat = y_hat.reshape(features.shape[0], self.num_classes, h // p, w // p)
        y_hat = interpolate(y_hat, size=(h, w), mode="bilinear")  # [B, C, H, W]

        return y_hat

    def custom_dice_loss(self, y_hat: torch.Tensor, batch: dict):
        print("Class mask shape: ", batch["class_mask"].shape)
        print("Y_hat shape: ", y_hat.shape)

        class_mask = batch["class_mask"].moveaxis(-1, 1)
        loss_fn = DiceLoss(
            mode="multilabel",
            classes=len(self.classes),
            from_logits=True,
        )

        loss = loss_fn(y_hat, class_mask)

        return loss

    def custom_binary_cross_entropy(self, y_hat: torch.Tensor, y, ignore_mask):
        print("Class mask shape: ", y.shape)
        print("Ignore mask shape: ", ignore_mask.shape)
        print("Y_hat shape: ", y_hat.shape)
        y_hat = y_hat.moveaxis(1, -1)
        print("Y_hat shape: ", y_hat.shape)
        loss = F.binary_cross_entropy_with_logits(
            y_hat,
            y,
            weight=(1 - ignore_mask),
        )
        return loss

    def focal_loss(self, y_hat, y, ignore_mask, alpha=0.25, gamma=2.0):
        """Focal loss for handling class imbalance"""
        y = y.permute(0, 3, 1, 2).float()
        ignore_mask = ignore_mask.permute(0, 3, 1, 2).float()

        bce_loss = F.binary_cross_entropy_with_logits(y_hat, y, reduction="none")
        pt = torch.exp(-bce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * bce_loss

        # Apply ignore mask
        valid_mask = 1 - ignore_mask
        focal_loss = focal_loss * valid_mask

        return focal_loss.sum() / valid_mask.sum()

    def train(self):
        """Linear model training loop"""
        self.classifier.train()
        optimizer = torch.optim.SGD(self.classifier.parameters(), lr=self.lr)
        val_frequency = 50  # Validate every 10 epochs
        for epoch in range(self.num_epochs):
            for batch in tqdm(self.patch_dataloader_train, desc="Training"):
                features, class_mask, ignore_mask = batch
                features = features.to(self.device)  # (B, total_channels, H, W)
                class_mask = class_mask.to(self.device)  # (B, H, W, C)
                ignore_mask = ignore_mask.to(self.device)

                # Direct prediction - no reshaping needed
                y_hat = self.classifier(features)  # (B, num_classes, H, W)

                # Ensure same size
                if y_hat.shape[2:] != class_mask.shape[1:3]:
                    y_hat = F.interpolate(
                        y_hat, size=class_mask.shape[1:3], mode="bilinear"
                    )

                loss = self.loss(y_hat, class_mask.float(), ignore_mask.float())
                loss = loss.mean()

                optimizer.zero_grad()
                # loss.backward()
                # optimizer.step()

            if epoch % val_frequency == 0:
                print(f"Epoch {epoch}, Loss: {loss.item()}")
                # Evaluate on train set
                print("Evaluating on training set...")
                self.evaluate(dataloader=self.patch_dataloader_train)
                # Evaluate on validation set
                print("Evaluating on validation set...")
                self.evaluate()
        self.trained = True
        print("Training complete.")
        print(f"Training finished with final loss: {loss.item()}")

    def evaluate(self, dataloader=None):
        """Evaluate the linear probing model."""

        print("Evaluating linear model...")
        self.iou_classwise.reset()
        classwise_ious = []

        dataloader = dataloader if dataloader is not None else self.patch_dataloader_val
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                if self.profile_time:
                    start_time = time.time()

                features, labels, ignore_mask = batch  # Unpack TensorDataset
                features = features.to(self.device)  # [B, Num_patches, Embed_dim]
                labels = labels.to(self.device)  # [B, H, W, C]

                class_mask = labels.permute(0, 3, 1, 2).long()  # [B, C, H, W]

                b, c, h, w = class_mask.shape
                y_hat = self.classifier(features)  # [B, C, H, W]

                if y_hat.shape[2:] != class_mask.shape[2:]:
                    y_hat = F.interpolate(y_hat, size=(h, w), mode="bilinear")

                y_hat_prob = torch.sigmoid(y_hat)  # Convert to probabilities [0, 1]
                y_hat_binary = (
                    y_hat_prob > self.threshold
                ).long()  # Convert to binary [0, 1]

                classwise_ious.append(self.iou_classwise(y_hat_binary, class_mask))
                print(f"Classwise IoUs: {classwise_ious[-1]}")

                if self.profile_time:
                    end_time = time.time()
                    print(f"Batch evaluation time: {end_time - start_time:.2f}s")

        # Compute mean IoU for each class
        mean_ious = {}
        for c in classwise_ious[0]:
            mean_ious[c] = np.mean(
                [iou[c].item() for iou in classwise_ious if iou[c] != 0]
            )

        print("Logistic Regression Mean IoUs:", mean_ious)
        return mean_ious

    def freeze_encoder(self):
        """Manually freeze encoder"""
        # Store current state
        self._encoder_was_training = self.encoder.training

        # Store original requires_grad state
        for name, param in self.encoder.named_parameters():
            self._original_requires_grad[name] = param.requires_grad
            param.requires_grad = False

        self.encoder.eval()
        return self

    def unfreeze_encoder(self):
        """Manually unfreeze encoder"""
        # Restore requires_grad state
        for name, param in self.encoder.named_parameters():
            if name in self._original_requires_grad:
                param.requires_grad = self._original_requires_grad[name]

        # Restore training mode
        if self._encoder_was_training is not None:
            self.encoder.train(self._encoder_was_training)

        return self


class PatchLinearClassifierCallback(Callback):

    def __init__(self, config):
        self.transform = A.Compose(
            [
                A.PadIfNeeded(
                    min_height=config.input_resolution[0],
                    min_width=config.input_resolution[1],
                    # Avoids reflective padding
                    border_mode=cv2.BORDER_CONSTANT,
                    value=(0, 0, 0),
                    p=1,
                ),
                A.CenterCrop(
                    config.input_resolution[0],
                    config.input_resolution[1],
                ),
            ]
        )
        self.eval_module = RetrievalDataModule(
            data_path=config.eval_data_path,
            batch_size=config.batch_size,
            classes=["wire", "ball", "wedge", "epoxy"],
            num_workers=config.num_workers,
            train_size=0.7,
            return_manufacturer=True,
            return_device=True,
            input_resolution=(512, 512),
            normalize=config.normalize,
        )

        self.eval_module.setup(stage="fit")
        self.train_loader = self.eval_module.train_dataloader()
        self.val_dataloader = self.eval_module.val_dataloader()
        self.classes = config.classes
        self.eval_frequency = config.eval_frequency

    def on_train_epoch_end(self, trainer, pl_module):
        linear_model = LinearPatchClassifier(
            encoder=pl_module.get_encoder(),
            classes=self.classes,
            train_loader=self.train_loader,
            val_loader=self.val_dataloader,
        )
        # Freeze for evaluation
        linear_model.freeze_encoder()

        try:
            results = linear_model.evaluate()
            mean_iou = sum(results.values()) / len(results)
            pl_module.log("val/linear_mean_iou", mean_iou)
        finally:
            # Always unfreeze, even if evaluation fails
            linear_model.unfreeze_encoder()
