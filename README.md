# 🍎 Fruit-360 Classification with VGG16 Transfer Learning

Fruit classification model achieving 100% accuracy on 24-class Fruits-360 test subset (3,110 images). Built with VGG16 transfer learning, fine-tuning, and professional evaluation pipeline.

## 🎯 Features
VGG16 Transfer Learning: Frozen base + custom head (512→256→24 classes)

Two-Phase Training: Head training (91% val) → Fine-tune top-8 layers (99.9% val)

Data Augmentation: Rotation, zoom, flips for robust generalization

Professional Pipeline: Confusion matrix, Top-k accuracy, prediction grids

24 Classes: 15 apple varieties + veggies (cabbage, carrot, cucumber, etc.)

## 📊 Performance

| Metric   | Train  | Val    | Test   |
|:--------:|:------:|:------:|:------:|
| Accuracy | 99.2%  | 99.8%  | 100.0% |
| Loss     | 0.03   | 0.01   | 0.008  |

![alt text](https://github.com/dawmro/fruit_classification/blob/main/cm.png?raw=true)


## 📁 Structure
```
fruit-classification/
├── fruit_classification.ipynb     # Complete pipeline
├── requirements.txt              # Dependencies
├── fruits-360-original-size/     # Dataset (downloaded via notebook)
|   └── fruits-360-original-size/
│       ├── Training/     # 6,231 imgs, 24 classes
│       ├── Validation/   # 3,114 imgs  
│       └── Test/         # 3,110 imgs
└── cm.png                       # Confusion matrix
```



## 🛠 Quick Start

1. Clone & Install
``` sh
git clone https://github.com/yourusername/fruit-classification.git
cd fruit-classification
pip install -r requirements.txt
```

2. Train & Evaluate
``` sh
jupyter notebook fruit_classification.ipynb
```

## 📈 Results Highlights
Perfect Test Accuracy: 100% on 3,110 test images

Robust: Handles lighting/color variations via augmentation

Fast Inference: ~20ms/image on CPU

Lightweight: 57MB model size

## 🔬 Technical Details
Dataset: Fruits-360 subset (24 classes, ~12k images)
​

Preprocessing: VGG16-specific + augmentation (rotation=20°, zoom=0.2)

Optimizer: Adam (1e-3 → 1e-5 fine-tune)

Callbacks: EarlyStopping(patience=5), ReduceLROnPlateau

Hardware: Trained on consumer CPU

## 🏆 Limitations & Future Work
Subset Only: 24/131 classes → Extend to full dataset

Input Size: 128×128 → Upgrade to 224×224 (+3-5% expected)

More Classes: Full Fruits-360 (131 classes)


## 📝 Citation
```
@misc{oltean2017fruits360,
  author = {Mihai Oltean},
  title = {Fruits-360 dataset},
  year = {2017-},
  howpublished = {\url{https://github.com/fruits-360/fruits-360-original-size}},
  note = {Accessed: 2026-03-08}
}
```
