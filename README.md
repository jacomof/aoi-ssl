# AOI-SSL

Official code release for our CVPRW 2026 (AI4RWC) paper:

AOI-SSL: Self-Supervised Framework for Efficient Segmentation of Wire-Bonded Semiconductors in Optical Inspection.

## Overview

AOI-SSL is a self-supervised learning and fine-tuning pipeline for industrial visual inspection. The repository includes:

- self-supervised pretraining components (MAE/DINO/iBOT-style modules),
- downstream segmentation models (ViT and FasterViT variants),
- retrieval-style evaluation utilities (kNN-based segmentation retrieval),
- configurable experiment setups under [configs](configs).

## Confidentiality And Public Release Scope

The original project was developed with a private industrial dataset. To protect confidentiality:

- private filesystem paths and internal endpoints have been removed,
- configuration paths use public placeholders,
- data loaders support regular image files instead of private HDF5-specific conventions,
- synthetic stub datasets can be generated locally for smoke testing.

No proprietary data is included in this repository.

## Repository Structure

- [configs](configs): pretraining and fine-tuning YAML configurations
- [data](data): Lightning data modules and pytorch datasets
- [segmentation](segmentation): segmentation models and ViT/FasterViT implementations
- [retrieval](retrieval): patch and image-based retrieval
- [pretrain](ssl): self-supervised training code
- [tests](tests): exploratory notebooks
- [tools](tools): utility scripts (including stub dataset generation)

## Quick Start

1. Install dependencies in your environment (PyTorch, Lightning, Albumentations, OpenCV, TorchMetrics, timm, etc.).
2. Generate MNIST stub dataset by running the download_mnist_preprocess.ipynb jupyter notebook
3. Point config files to your dataset/checkpoint paths if needed.
4. Run your chosen pretraining or fine-tuning entrypoint.

## Stub Dataset Layout

The default stub ImageNet-based dataset layout is:

```text
datasets/MNIST
	pretrain/
		train/
			sample_buffer0.png
			sample_buffer1.png
		val/
			sample_buffer0.png
			sample_buffer1.png
	finetune/
		train/
			image/
				sample_buffer0.png
				sample_buffer1.png
			lbl/
				sample.png
		val/
			image/
				sample_buffer0.png
				sample_buffer1.png
			lbl/
				sample.png
		test/
			image/
				sample_buffer0.png
				sample_buffer1.png
			lbl/
				sample.png
```

Mask convention:

- single-channel label maps with integer class IDs,
- classes are configured via `classes` in YAML (default: wire/ball/wedge/epoxy).

## Citation

If you use this repository, please cite the paper once the workshop proceedings are available.
