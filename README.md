# Detection of Calving Events at Perito Moreno Glacier Using Deep Learning on Time-Lapse Imagery

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-%3E%3D2.10-orange.svg)](https://www.tensorflow.org/)
[![DOI](https://img.shields.io/badge/DOI-[to%20be%20filled]-lightgrey)](https://doi.org/)

Official code and data repository for the paper:

> **Detection of Calving Events at Perito Moreno Glacier Using Deep Learning on Time-Lapse Imagery**  
> Catalina Zapata, Angelica Monserrat Buenrostro, Sergio Ruiz, Carlos F. Navarro, Leoncio Cabrera, Edgardo Casanova, Gino Casassa, Masahiro Minowa  
> *[Journal name — to be filled]*  
> DOI: [to be filled upon acceptance]

---

## Abstract

Calving is a major contributor to ice mass loss in lake- and tidewater-terminating glaciers, yet its episodic nature makes systematic detection difficult. We present an automated approach for classifying calving events at the Brazo Rico front of the Perito Moreno Glacier using terrestrial time-lapse imagery. The dataset consists of approximately 2,900 labeled hourly optical images acquired between 21 November 2018 and 29 May 2019. We developed a lightweight two-dimensional convolutional neural network (2D-CNN) and compared its performance with three widely used deep learning architectures: VGG16, ResNet-50, and MobileNet. The proposed 2D-CNN achieved an accuracy of 86.2% and an F1-score of 0.865, comparable to VGG16 and MobileNet, while requiring a model size of only 0.487 MB and inference times 3 to 9 times faster than compared architectures — enabling deployment on low-cost embedded hardware for sustained autonomous operation at remote glacier monitoring stations.

---

## Key Results

*
*
*

---

## Repository Structure

```
.
├── README.md
├── CNN_Calving_Model.ipynb          # Lightweight 2D-CNN — trained from scratch
├── MobileNet_Calving_Model.ipynb    # MobileNet — frozen backbone, transfer learning
├── VGG16_Calving_Model.ipynb        # VGG16 — frozen backbone, transfer learning
├── ResNet50_Calving_Model.ipynb     # ResNet-50 — frozen backbone, transfer learning
├── data/                            # Dataset directory (see Dataset section)
│   ├── train_images.npy
│   ├── train_labels.npy
│   ├── val_images.npy
│   ├── val_labels.npy
│   ├── test_images.npy
│   └── test_labels.npy
└── outputs/                         # Generated at runtime
    ├── model_calving.keras
    ├── VGG16_calving.keras
    ├── ResNet50_calving.keras
    ├── MNet_calving.keras
    ├── history_*.npy
    ├── y_pred_*.npy
    ├── training_curves*.png
    ├── confusion_matrices*.png
    └── sample_predictions*.png
```

---

## Dataset

Time-lapse images were acquired with a fixed Canon EOS 50D camera (28 mm lens, f/2.8, ISO 100) installed approximately 1,300 m from the Brazo Rico front of Perito Moreno Glacier (50.5°S, 73.1°W), capturing one frame per hour between 21 November 2018 and 29 May 2019.

**Preprocessing pipeline:**
1. Quality filtering: night-time frames and images with poor visibility removed (4,500 → 3,114 usable images)
2. Binary labeling: calving / no-calving (1,453 calving images identified)
3. Region-of-interest (ROI) selection: proglacial lake segmented with [SAM 2](https://github.com/facebookresearch/segment-anything-2), followed by manual selection of the most active calving sector
4. Resizing to 256 × 256 pixels and intensity normalization
5. Dataset balanced to ~2,900 images (equal class representation)
6. Random split: **80% training / 10% validation / 10% test** (fixed seed for reproducibility)

Data are stored as pre-normalized NumPy arrays of shape `(N, 256, 256, 3)` with pixel values in `[0, 1]`. Labels are binary arrays (`1` = calving, `0` = no-calving).

**Dataset archive:** [DOI / repository link — to be filled upon publication]

---

## Requirements

```
python >= 3.9
tensorflow >= 2.10
numpy
matplotlib
seaborn
scikit-learn
```

Install all dependencies:

```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn
```

The notebooks were trained on an **NVIDIA A100 GPU**. A GPU is recommended for VGG16 and ResNet-50; the 2D-CNN can be trained on CPU in reasonable time.

---

## Usage

**1. Clone this repository**

```bash
git clone https://github.com/[username]/[repo-name].git
cd [repo-name]
```

**2. Download the dataset**

Download the `.npy` files from [dataset archive — DOI to be filled] and place them in the `data/` folder:

```
data/
├── train_images.npy
├── train_labels.npy
├── val_images.npy
├── val_labels.npy
├── test_images.npy
└── test_labels.npy
```

**3. Run a notebook**

Open any of the four notebooks and run all cells in order. The only configuration required is in the **Section 0 cell** at the top of each notebook:

```python
DATA_DIR   = "data"     # path to folder with .npy files
OUTPUT_DIR = "outputs"  # path for saving models, predictions, and figures
```

Each notebook is self-contained and follows the same structure:
data loading → model definition → training → evaluation → confusion matrices → model saving.

---

## Reproducibility

- The dataset split is fixed (`random_state=1`) and identical across all four notebooks.
- The decision threshold is selected to maximise F1-score on the **validation set** and applied without modification to the test set. The optimal threshold is printed at runtime and saved alongside the predictions.
- All training curves, confusion matrices, and sample prediction figures are saved automatically to `outputs/`.
- Pre-trained backbone weights for VGG16, ResNet-50, and MobileNet are downloaded automatically from TensorFlow/Keras (ImageNet weights).

---

## Citation

If you use this code or dataset in your work, please cite:

```bibtex
@article{zapata2025calving,
  author  = {Zapata, Catalina and Buenrostro, Angelica Monserrat and Ruiz, Sergio and
             Navarro, Carlos F. and Cabrera, Leoncio and Casanova, Edgardo and
             Casassa, Gino and Minowa, Masahiro},
  title   = {Detection of Calving Events at {Perito Moreno} Glacier Using Deep Learning
             on Time-Lapse Imagery},
  journal = {[Journal name]},
  year    = {[Year]},
  doi     = {[DOI]}
}
```

---

## Related Work

This repository complements the multimodal calving monitoring framework at Perito Moreno Glacier. For seismic-based detection using template matching at the same site, see:

> Cabrera, L., et al. (2026). Localized calving sensitivity at Perito Moreno Glacier (Patagonia) revealed by seismic template matching and time-lapse images. *Journal of Geophysical Research: Earth Surface*, 131(4), e2025JF008688. https://doi.org/10.1029/2025JF008688

---

## License

- **Code** (notebooks): MIT License
- **Dataset** (.npy files): CC BY 4.0 — free to use provided the associated paper is cited.

---

## Contact

For questions about the code or dataset, please open an issue or contact Catalina Zapata at catalina.zapata@ug.uchile.cl.
