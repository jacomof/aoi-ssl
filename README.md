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
- [data](data): Lightning data modules
- [segmentation](segmentation): segmentation models and compatibility modules
- [retrieval](retrieval): kNN retrieval-based evaluation utilities
- [ssl](ssl): copied self-supervised training code from internal experiments
- [tests](tests): exploratory notebooks
- [tools](tools): utility scripts (including stub dataset generation)

## Quick Start

1. Install dependencies in your environment (PyTorch, Lightning, Albumentations, OpenCV, TorchMetrics, timm, etc.).
2. Generate synthetic stub datasets:

```bash
python tools/create_stub_datasets.py --output datasets
```

3. Point config files to your dataset/checkpoint paths if needed.
4. Run your chosen pretraining or fine-tuning entrypoint.

## Stub Dataset Layout

The default public-friendly image-based layout is:

```text
datasets/
	pretrain_with_unlabelled/
		train/*.png
		val/*.png
		test/*.png
	pretrain_split/
		train/sample_0000.png
		train/sample_0000_mask.png
		val/sample_0000.png
		val/sample_0000_mask.png
		test/sample_0000.png
		test/sample_0000_mask.png
	semantic_split/
		train/*.png + *_mask.png
		val/*.png + *_mask.png
		test/*.png + *_mask.png
```

Mask convention:

- single-channel label maps with integer class IDs,
- classes are configured via `classes` in YAML (default: wire/ball/wedge/epoxy).

## Notes On Compatibility

This release includes compatibility shims for legacy `segmentation.*` import paths used in internal notebooks and scripts. These shims are intended to keep common workflows importable in the cleaned public structure.

## Citation

If you use this repository, please cite the paper once the workshop proceedings are available.
