# 🧘 AOI-SSL 📟

Official code release for our CVPRW 2026 (AI4RWC) paper:

AOI-SSL: Self-Supervised Framework for Efficient Segmentation of Wire-Bonded Semiconductors in Optical Inspection.

## 🔍 Overview

AOI-SSL is a self-supervised learning and fine-tuning pipeline for industrial visual inspection. The repository includes:

- self-supervised pretraining components (MAE/DINO/iBOT-style modules),
- downstream segmentation models (ViT and FasterViT variants),
- retrieval-style evaluation utilities (kNN-based segmentation retrieval),
- configurable experiment setups under [configs](configs),
- experiment tracking using MLFlow.

## 🔒 Confidentiality And Public Release Scope

The original project was developed with a private industrial dataset. To protect confidentiality:

- private filesystem paths and internal endpoints have been removed,
- configuration paths use public placeholders,
- data loaders support regular image files instead of private HDF5-specific conventions,
- synthetic stub datasets can be generated locally for smoke testing.

No proprietary data is included in this repository.

## ⚖️ Licensing And Third-Party Notices

This repository is distributed under Apache-2.0, with the exception of specific third-party files that keep their original upstream license terms.

In particular, parts of `segmentation/models/faster_vit` are derived from NVIDIA FasterViT code and include NVIDIA copyright and license headers. Those files are not relicensed by this repository and remain subject to their original terms, including non-commercial and other use restrictions.

See `THIRD_PARTY_NOTICES` for a file-by-file provenance and licensing summary.

## 📁 Repository Structure

- [configs](configs): pretraining and fine-tuning YAML configurations
- [data](data): Lightning data modules and pytorch datasets
- [segmentation](segmentation): segmentation models and ViT/FasterViT implementations
- [retrieval](retrieval): patch and image-based retrieval
- [pretrain](ssl): self-supervised training code
- [tests](tests): exploratory notebooks showing usage and hyper-parameter tuning process for retrieval
- [scripts](scripts): scripts to launch pre-training and fine-tuning experiments and guide the use of the pretrain.py and train.py

## 🚀 Quick Start

1. Install dependencies in your environment (PyTorch, Lightning, Albumentations, OpenCV, TorchMetrics, timm, etc.).
2. Generate MNIST stub dataset by running the download_mnist_preprocess.ipynb jupyter notebook
3. Create or adapt config files (see existing [configs](configs)). Point config files to your dataset/checkpoint paths if needed.
4. Run your chosen pretraining or fine-tuning entrypoint ([fine-tuning](segmentation/train.py), [MAE-pretraining](pretrain/mae/pretrain.py), etc.). You can use or adapt existing experiments in [scripts](scripts) directly.

## 📊 Stub Dataset Layout

The default stub MNIST dataset layout is:

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

## 📖 Citation

If you use this repository, please cite the paper once the workshop proceedings are available.

## 🤝 Acknowledgements

This repository is built on top of existing [ViT and DINO implementations](https://github.com/facebookresearch/dino/tree/main) by M. Caron et al. The patch-level retrieval strategies are built on top of the [HummingBird implementation](https://github.com/vpariza/open-hummingbird-eval/tree/main) by V. Pariza. The FasterViT implementation was adapted from the [original NVIDIA repository](https://github.com/NVlabs/FasterViT) by A. Hatamizadeh et al. We thank the authors of these repositories for creating such high-quality and reproducible code.
