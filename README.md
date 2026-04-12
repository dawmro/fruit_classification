# 🍎 Fruit-360 Classification with VGG16 Transfer Learning

[![Full Test](https://img.shields.io/badge/Full%20141cl-93%25-orange)](https://github.com/dawmro/fruit_classification) [![Subset](https://img.shields.io/badge/24cl-100%25-brightgreen)](https://github.com/dawmro/fruit_classification)

Fruit classification system built with Transfer Learning using VGG16 pre-trained on ImageNet.

**93.0% accuracy** on **full Fruits-360** (141 classes, 24,810 test images). 

![Full Confusion Matrix](cm_full.png)

## 🎯 Features & Progress

| Version     | Classes | Train  | **Test** | Notebook |
|-------------|---------|--------|----------|--------------|
| **v2.0 Full** | **141** | **98.8%** | **93%** | [Full →](fruit_classification_full_dataset.ipynb) |
| v1.0 Subset       | 24      | 99.2%  | **100%** |  [Subset →](fruit_classification.ipynb) |

- **Custom Head**: (512→256→141 classes)
- **VGG16 Transfer Learning**: Frozen → Fine-tune top-8 layers
- **Two-Phase Training**: Head (82.8%) → Full (98.8% train)
- **Augmentation**: Rotation, zoom, flips for robust generalization
- **Professional Pipeline**: Auto-download, Confusion matrix, Top-k accuracy, prediction grids


## 📁 Structure
```
fruit-classification/
├── fruit_classification_full_dataset.ipynb  # 141 classes (93.0%)
├── fruit_classification.ipynb     # 24 classes (100%)
├── requirements.txt              # Dependencies
├── fruits-360-original-size/     # Dataset (downloaded via notebook)
|   └── fruits-360-original-size/
│       ├── Training/   # 48k imgs, 141 classes
│       ├── Validation/ # 24k imgs 
│       └── Test/       # 24k imgs
├── cm.png                       # Confusion matrix for 24 classes
└── cm_full.png      # Full dataset Confusion matrix for 141 classes
```



## 🛠 Quick Start
1. Create new virtual env:
``` sh
py -3.10 -m venv env
```
2. Activate your virtual env:
``` sh
env/Scripts/activate
```
3. Clone repo
``` sh
git clone https://github.com/dawmro/fruit_classification.git
cd fruit_classification
```
4. Install requirements
```sh
pip install -r requirements.txt
```
5. Run notebook
```sh
# Full 141 classes (93%)
jupyter notebook fruit_classification_full_dataset.ipynb
```

## 🔮 Model Architecture
```
VGG16 (14.7M frozen → 1.7M tuned) 
→ GlobalAvgPool → Dense(512+BN+Drop0.5) 
→ Dense(256+BN+Drop0.3) → Dense(141, softmax)
Total: 15.1M params (57MB)
```

## 📈 Results Highlights
Perfect Test Accuracy: 100% on 3,110 test images

Fast Inference: ~20ms/image on CPU

Lightweight: 57MB model size


## 🔬 Technical Details

Dataset: Fruits-360 full (141 classes)

Preprocessing: VGG16-specific + augmentation (rotation=20°, zoom=0.2)

Optimizer: Adam (1e-4 → 1e-5 fine-tune)

Callbacks: EarlyStopping(patience=5), ReduceLROnPlateau(patience=3)

Hardware: Trained on consumer GPU

![Full Confusion Matrix](class_distributions.png)

## 🏆 Limitations & Future Work
Find out why validation set gives only 22% accuracy.

Input Size: 128×128 → Upgrade to 224×224 (+3-5% expected)


## 📝 Citation
```
@misc{oltean2017fruits360,
  author = {Mihai Oltean},
  title = {Fruits-360 dataset},
  year = {2017-},
  howpublished = {\url{https://github.com/fruits-360/fruits-360-original-size}},
  note = {Accessed: 2026-04-12}
}
```
