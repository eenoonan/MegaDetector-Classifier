# MegaDetector-Classifier: Giraffes vs Zebras
## A derivative inquiry into imagery elements driving classification
### STAT 5720 Final Project

Project GitHub: 

Note: This has been appended to the original README from the MegaDetector-Classifier repo README. 
The original README content remains at the end of this document for historical and reference purposes.

### Objective Summary
The academic exercise objective is to take an existing code base and develop a novel application or extension.
The objective of this project is to take a dataset not included in the available models and use it to train a model 
based on the MegaDetector-Classifier code that will be used to examine a group of datasets that have been modified 
to emphasize specific imagery characteristics in an effort to identify the significance of those characteristics to 
the classification process and algorithm.

### Project Summary 
This effort used the MegaDetector-Classifier repo [1] to train a model using a data set of zebra and giraffe images taken 
from a photographic animal census conducted at Nairobi National Park in Kenya in 2015 [2]. As the images from this dataset 
included cropping coordinates for animals to align with the assigned classification and identification, the MegaDetector 
identification and cropping tool was not implemented to generate cropped images. Half of the dataset was used to train 
and test the model and the other half of the dataset was used to create three variations for evaluating the model's 
classification performance with imagery modification. The variations were: baseline (no change), silhouette (only black
outline on white background), and swatch (a computer extracted square of the selected animal's skin/fur pattern). 

### Installation and Execution Instructions
Follow the steps and example commands below to copy and run this code.

1. git clone https://github.com/eenoonan/MegaDetector-Classifier

2. Execute all steps in Jupyter notebook image_extract_process.ipynb
    
    This will download and unzip the Zebra and Giraffe data set as well as divide the set into two groups - 
one for training, test, and validation, and one for investigating imagery characteristics. The cropped images 
documented in the included annotations file will be extracted, binned, and documented in annotations files.


3. If a small dataset is desired for checking that code is functioning, run make_dev_subset.py:

    _python make_dev_subset_v1.py --src ./data/gzgc_dataset/dev_crops/cropped_resized --img-root ./data/gzgc_dataset/dev_crops/cropped_resized \
    --dst ./data/gzgc_dataset/dev_crops_subset/cropped_resized --class-col classification --label-col label --layout flat \
    --train-per-class 20 --val-per-class 8 --test-per-class 8_


4. Execute animal_batch.py to generate three datasets for imagery characteristic evaluations:

    _python animal_batch.py --input ./data/gzgc_dataset/eval_crops/cropped_resized \
    --annot ./data/gzgc_dataset/eval_crops/annot_eval_crop.csv --swatch-size 100 --erode 15_


5. Review configs/config.yaml to verify settings for dataset locations and balancing requirements


6. Execute main.py to train model based on the training portion of the imagery dataset:

    _python main.py --config configs/config.yaml_


7. Execute run_predict.py to evaluate desired datasets with trained model. Check/update checkpoint path and name.

    _python run_predict.py  --base-config ./configs/config.yaml  --ckpt ./weights/Crop/Plain/\
Crop_Res18_plain_071824-0-epoch=00-valid_mac_acc=100.00.ckpt  --datasets siteA=./data/gzgc_dataset/eval_crops/ cropped_resized/baseline \
siteB=./data/gzgc_dataset/eval_crops/cropped_resized/sil siteC=./data/gzgc_dataset/eval_crops/cropped_resized/swatch \
siteD=./data/gzgc_dataset/eval_crops/cropped_resized/bl_selected    \s
siteE=./data/gzgc_dataset/eval_crops/cropped_resized/sil_selected siteF=./data/gzgc_dataset/eval_crops/cropped_resized/swatch_selected \
--out-dir ./eval_results_


8. Execute find_missed.py to generate list of misclassified images for each dataset evaluated:
    
    _python find_misses.py --config eval_results/config_siteA.yaml \
--ckpt ./weights/Crop/Plain/Crop_Res18_plain_071824-0-epoch=00-valid_mac_acc=100.00.ckpt --name siteA_


9. Execute plot_metrics.py to generate visual summaries of the classification performance for each dataset evaluated.
    _python plot_metrics.py miss_results/predictions_siteA.csv miss_results/predictions_siteB.csv \
miss_results/predictions_siteC.csv miss_results/predictions_siteD.csv miss_results/predictions_siteE.csv miss_results/predictions_siteF.csv_

### Limitations
This code has been specifically tailored for the selected imagery dataset annotations and modifying those annotations to 
satisfy the MegaDetector-Classifier annotation requirements. 

### References
[1] @misc{hernandez2024pytorchwildlife,
      title={Pytorch-Wildlife: A Collaborative Deep Learning Framework for Conservation},
      author={Andres Hernandez and Zhongqi Miao and Luisa Vargas and Sara Beery and Rahul Dodhia and Juan Lavista},
      year={2024},
      eprint={2405.12930},
      archivePrefix={arXiv},
}


[2]
Parham J, Crall J, Stewart C, Berger-Wolf T, Rubenstein DI. Animal population censusing at scale with citizen science 
and photographic identification. In AAAI Spring Symposium-Technical Report 2017 Jan 1.
(https://lila.science/datasets/great-zebra-giraffe-id)

### Licenses
The following licenses apply to the resources implemented in this project. 

MegaDetector-Classifier: 

MIT License

Copyright (c) 2026 Microsoft

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Great Zebra and Giraffe Count and ID via Labeled Information Library of Alexandria: Biology and Conservation:

Community Data License Agreement (permissive variant)
(https://cdla.dev/permissive-1-0/)


See original README content below.
____________________________________________________________

# MegaDetector-Classifier
Microsoft AI for Good Lab's open-source classification fine-tuning tool — train custom species classifiers on your own camera-trap datasets and deploy them through PyTorch-Wildlife.

[![License](https://img.shields.io/github/license/microsoft/MegaDetector-Classifier)](https://github.com/microsoft/MegaDetector-Classifier/blob/main/LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch-Wildlife](https://img.shields.io/badge/PyTorch--Wildlife-ecosystem-green.svg)](https://github.com/microsoft/Biodiversity)

MegaDetector-Classifier is part of the [microsoft/Biodiversity](https://github.com/microsoft/Biodiversity) ecosystem and is powered by the [PyTorch-Wildlife](https://github.com/microsoft/PytorchWildlife) framework. It is free, open-source, and available under the MIT license.

## Part of the Biodiversity Ecosystem

MegaDetector-Classifier is one tool in a larger open-source ecosystem from the Microsoft AI for Good Lab. Each project lives in its own repository, with the [microsoft/Biodiversity](https://github.com/microsoft/Biodiversity) umbrella tying them together.

| Repository | Description |
|---|---|
| [microsoft/Biodiversity](https://github.com/microsoft/Biodiversity) | The umbrella repository — documentation hub for the AI for Good Lab's biodiversity work |
| [microsoft/MegaDetector](https://github.com/microsoft/MegaDetector) | Animal, human, and vehicle detection for camera-trap images |
| [microsoft/PytorchWildlife](https://github.com/microsoft/PytorchWildlife) | The collaborative deep learning framework that hosts MegaDetector, species classifiers, and demo notebooks |
| [microsoft/MegaDetector-Acoustic](https://github.com/microsoft/MegaDetector-Acoustic) | Bioacoustic AI for audio-based wildlife detection and classification |
| [microsoft/MegaDetector-Overhead](https://github.com/microsoft/MegaDetector-Overhead) | Wildlife detection in aerial and drone imagery |
| [microsoft/MegaDetector-Sonar](https://github.com/microsoft/MegaDetector-Sonar) | Sonar-based wildlife detection for aquatic monitoring |
| [microsoft/MegaDetector-Classifier](https://github.com/microsoft/MegaDetector-Classifier) | **This repo** — classification fine-tuning for camera-trap species identification |
| [microsoft/SPARROW](https://github.com/microsoft/SPARROW) | Solar-Powered Acoustic and Remote Recording Observation Watch — AI-enabled edge device for field recording |

## Overview

MegaDetector-Classifier is a training toolkit for fine-tuning ResNet-based species classifiers on custom camera-trap image datasets. The output weights integrate directly with the [PyTorch-Wildlife](https://github.com/microsoft/PytorchWildlife) framework, making it straightforward to deploy a classifier trained on your own data.

**Key capabilities:**
- ResNet-18 and ResNet-50 classifier training using PyTorch Lightning
- Three data-splitting strategies designed for camera-trap realities: random, location-based, and sequence-based
- YAML-based configuration — no code changes required for most use cases
- Demo data included for immediate testing without your own dataset

**Designed for:**
- Conservation practitioners adapting existing classifiers to new geographic regions
- Researchers adding new species to the PyTorch-Wildlife model zoo
- Projects running MegaDetector detection upstream and needing a matched classifier downstream

## Installation

### Using pip

```bash
git clone https://github.com/microsoft/MegaDetector-Classifier
cd MegaDetector-Classifier
pip install -r requirements.txt
```

### Using conda

```bash
git clone https://github.com/microsoft/MegaDetector-Classifier
cd MegaDetector-Classifier
conda env create -f environment.yaml
conda activate PT_Finetuning
```

**Requirements:** Python 3.9+

## Quick Start

1. Configure `configs/config.yaml` — set `dataset_root`, `annotation_dir`, `num_classes`, and `split_type`
2. Run training:

```bash
python main.py
```

Output weights are saved to the `weights/` directory and can be loaded directly into PyTorch-Wildlife.

## Data Preparation

### Data Structure

Images should be stored in a single flat directory (no nested subdirectories). An `annotations.csv` file — placed outside the images directory — maps each image to its class:

```plaintext
MegaDetector-Classifier/
├── data/
│   ├── imgs/                       # All images stored here (flat)
│   └── annotation_example.csv      # Annotations file
└── configs/config.yaml
```

### Annotation File Format

The CSV must contain three columns:

| Column | Description | Example |
|---|---|---|
| `path` | Relative path to the image | `imgs/leopard_001.jpg` |
| `classification` | Integer class ID | `0` |
| `label` | Human-readable class name | `leopard` |

### Data Splitting

MegaDetector-Classifier supports three splitting strategies, selected via `split_type` in `config.yaml`:

| Strategy | When to use | Extra column required |
|---|---|---|
| `random` | Balanced class distribution; not recommended for camera-trap bursts | None |
| `location` | Keeps all images from one camera location in the same split | `Location` |
| `sequence` | Groups burst images within 30-second windows before splitting | `Photo_time` (YYYY-MM-DD HH:MM:SS) |

> **Camera-trap note:** Random splitting is not recommended because burst images of the same animal can appear in both training and validation sets, causing artificially high validation accuracy. Use `location` or `sequence` splitting instead.

### Demo Data

Download demo data to test the pipeline without your own dataset:

```bash
# Download and extract
wget https://zenodo.org/records/15376499/files/demo_data_clf.zip
unzip demo_data_clf.zip -d data/
```

Then set `dataset_root: ./data/imgs` in `configs/config.yaml` and run `python main.py`.

## Loading PytorchWildlife classifiers

In addition to training a custom classifier, MegaDetector-Classifier ships a PyTorch-Wildlife–compatible inference layer. Pretrained species classifiers can be loaded with a single import — the same `classification` subpackage layout that PyTorch-Wildlife exposes:

```python
from src.models import classification as pw_classification

# Pick any of the available loaders below
model = pw_classification.AI4GAmazonRainforest(device="cuda")
results = model.single_image_classification("path/to/image.jpg")
```

Available loaders:

- `AI4GAmazonRainforest` — Amazon Rainforest species classifier (ResNet-50, v1/v2 weights)
- `AI4GOpossum` — binary opossum / not-opossum classifier (ResNet-50)
- `AI4GSnapshotSerengeti` — Snapshot Serengeti 10-class classifier (ResNet-18)
- `CustomWeights` — load your own ResNet weights (e.g. those produced by `main.py`)
- `DeepfauneClassifier` — Deepfaune ViT-L/14 DINOv2 classifier (timm backbone)
- `DFNE` — DFNE ViT-L/14 DINOv2 classifier (timm backbone)
- `SpeciesNetTFInference` — SpeciesNet TensorFlow classifier (optional, see below)

### Optional: SpeciesNet

`SpeciesNetTFInference` depends on the `speciesnet` package, which pulls in TensorFlow. It is **not** installed by `requirements.txt` / `environment.yaml`. To enable it:

```bash
pip install speciesnet
```

If `speciesnet` is not installed, `from src.models import classification` still succeeds — only instantiating `SpeciesNetTFInference()` raises an `ImportError` with installation instructions.

## Repository Structure

```
MegaDetector-Classifier/
├── main.py                          # Training entry point
├── requirements.txt                 # pip dependencies
├── environment.yaml                 # conda environment
├── configs/
│   └── config.yaml                  # Training configuration
└── src/
    ├── algorithms/
    │   └── plain.py                 # Training algorithm (PyTorch Lightning)
    ├── data/                        # Inference data layer (PyTorch-Wildlife-compatible)
    │   ├── transforms.py            # Classification / MegaDetector transforms
    │   └── datasets.py              # ImageFolder / DetectionCrops loaders
    ├── datasets/
    │   └── custom.py                # Custom training dataset loader
    ├── models/
    │   ├── plain_resnet.py          # ResNet-18/50 training classifier
    │   └── classification/          # Inference classifiers (PyTorch-Wildlife-compatible)
    │       ├── base_classifier.py
    │       ├── resnet_base/         # AI4GAmazon, AI4GOpossum, Serengeti, CustomWeights
    │       ├── timm_base/           # DeepfauneClassifier, DFNE
    │       └── speciesnet_base/     # SpeciesNetTFInference (optional)
    └── utils/
        ├── batch_detection_cropping.py   # MegaDetector crop integration
        ├── data_splitting.py             # Random / location / sequence splits
        └── utils.py                      # Shared utilities
```

## Citation

If you use MegaDetector-Classifier in your research, please cite the PyTorch-Wildlife paper:

```bibtex
@misc{hernandez2024pytorchwildlife,
      title={Pytorch-Wildlife: A Collaborative Deep Learning Framework for Conservation},
      author={Andres Hernandez and Zhongqi Miao and Luisa Vargas and Sara Beery and Rahul Dodhia and Juan Lavista},
      year={2024},
      eprint={2405.12930},
      archivePrefix={arXiv},
}
```

You can also cite this software directly using the [`citation.cff`](citation.cff) file in this repository.

## Contributing

Issues, feature requests, and pull requests are welcome at [microsoft/MegaDetector-Classifier/issues](https://github.com/microsoft/MegaDetector-Classifier/issues).

For framework-level changes (PyTorch-Wildlife API, models, datasets), see [microsoft/PytorchWildlife](https://github.com/microsoft/PytorchWildlife). For ecosystem-wide questions, see the [microsoft/Biodiversity](https://github.com/microsoft/Biodiversity) umbrella.
