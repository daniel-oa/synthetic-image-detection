
# Synthetic Image Detection

A comparative deep learning study investigating the classification of real and AI-generated images using a custom Convolutional Neural Network (CNN) and MobileNetV2 transfer learning.

## Overview

The increasing accessibility of generative AI tools has made it easier to produce highly realistic synthetic images. Distinguishing AI-generated images from authentic photographs is becoming an increasingly relevant computer vision problem.

This project explores how different deep learning approaches perform on a binary image classification task: identifying whether an image is **FAKE (AI-generated)** or **REAL**.

Three approaches are implemented and evaluated:

1. **Custom CNN:** A convolutional neural network trained from scratch.
2. **MobileNetV2 Feature Extraction:** A pretrained MobileNetV2 model used as a frozen feature extractor with a new classification head.
3. **MobileNetV2 Fine-Tuning:** The pretrained network is partially unfrozen and further trained to adapt its learned features to the classification task.

The objective is to compare these approaches and investigate the effect of transfer learning and fine-tuning on classification performance.

## Dataset

The project uses the **CIFAKE: Real and AI-Generated Synthetic Images** dataset.

Dataset source: [CIFAKE on Kaggle](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)

The dataset contains:

| Split | FAKE | REAL | Total |
|---|---:|---:|---:|
| Training | 50,000 | 50,000 | 100,000 |
| Testing | 10,000 | 10,000 | 20,000 |

The images are originally 32 × 32 RGB images.

For the MobileNetV2 experiments, images are resized to 96 × 96 to accommodate the network's input requirements.

The training set is further divided into training and validation subsets, using an 80:20 split.

**Class labels:**
- FAKE: AI-generated images
- REAL: Real images

## Methodology

### 1. CNN Baseline

A custom convolutional neural network is trained from scratch to establish a baseline for the classification task.

The architecture consists of:
- Two convolutional layers with ReLU activation.
- Max-pooling layers for spatial downsampling.
- A flattening layer.
- A fully connected hidden layer.
- A sigmoid output layer for binary classification.

The model is trained on the original 32 × 32 images for 10 epochs.

### 2. MobileNetV2 Feature Extraction

MobileNetV2, pretrained on ImageNet, is used as the foundation for transfer learning.

Its original classification head is removed and replaced with a new classification head consisting of:
- Global Average Pooling.
- Dropout.
- A sigmoid output layer.

The pretrained MobileNetV2 base is frozen during training. Consequently, only the newly added classification head is trained.

This approach evaluates how effectively pretrained visual features can be applied to the synthetic image detection task without modifying the original network weights.

### 3. MobileNetV2 Fine-Tuning

The feature extraction model is further developed through fine-tuning.

The MobileNetV2 base is unfrozen, but only its final layers are allowed to train. Earlier layers remain frozen, and Batch Normalization layers are kept frozen.

A lower learning rate is used to allow the network to gradually adapt its pretrained features to the target dataset.

This experiment investigates whether adapting deeper pretrained features improves classification performance.

## Results

All three approaches are evaluated on the CIFAKE test set containing 20,000 images.

### Model Comparison

| Metric | CNN Baseline | MobileNetV2 Feature Extraction | MobileNetV2 Fine-Tuned |
|---|---:|---:|---:|
| Test Accuracy | 92.94% | 89.24% | **95.96%** |
| Test Loss | 0.2104 | 0.2626 | **0.1402** |
| Macro F1-Score | 93% | 89% | **96%** |

### Performance Comparison

The fine-tuned MobileNetV2 model achieved the highest test accuracy of **95.96%**, outperforming:

- The CNN baseline by 3.02 percentage points.
- MobileNetV2 feature extraction by 6.72 percentage points.

Interestingly, MobileNetV2 feature extraction performed below the custom CNN baseline. However, fine-tuning significantly improved its performance.

This suggests that adapting pretrained features to the target classification task was important for achieving better results.

### Fine-Tuning Observations

During fine-tuning, training accuracy increased to approximately 99.88%, while validation accuracy remained around 96%.

Validation loss reached its minimum at epoch 5 before increasing, indicating signs of overfitting as training continued.

Despite this, the final fine-tuned model achieved 95.96% test accuracy.

## Visualisations

The notebooks include visualisations covering:

- Dataset samples and class distribution.
- Training and validation accuracy.
- Training and validation loss.
- Model performance comparisons.

These visualisations provide insight into model learning behaviour and the differences between the three approaches.

## Technologies Used

- **Python** — Programming language.
- **TensorFlow / Keras** — Deep learning model development and training.
- **MobileNetV2** — Pretrained convolutional neural network.
- **NumPy** — Numerical operations.
- **Pandas** — Results organisation and comparison.
- **Matplotlib** — Data visualisation.
- **Jupyter Notebook** — Experiment development and documentation.

## Repository Structure

```text
synthetic-image-detection/
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   └── 02_mobilenetv2_transfer_learning.ipynb
│
├── models/
│   ├── cnn_baseline.keras
│   ├── mobilenetv2_feature_extraction.keras
│   └── mobilenetv2_fine_tuned.keras
│
├── .gitignore
└── README.md
```

The dataset is not included in the repository. It must be downloaded separately from Kaggle and placed in the `dataset/archive/` directory, with the `train/` and `test/` folders inside it.

## Limitations

- **Dataset dependency:** Evaluation was performed exclusively on CIFAKE. Performance on other datasets and image generators has not been established.
- **Image resolution:** MobileNetV2 experiments use images resized from 32 × 32 to 96 × 96. Resizing does not restore image details lost at the original resolution.
- **Overfitting:** The fine-tuned model showed a considerable difference between training and validation accuracy.
- **Generalisation:** The experiment does not establish robustness against newer image-generation techniques, image compression, or post-processing.
- **Deployment:** Inference speed and computational requirements were not benchmarked.

## Conclusion

This project compares a custom CNN trained from scratch with two transfer learning approaches using MobileNetV2.

While the frozen MobileNetV2 model did not outperform the CNN baseline, fine-tuning its deeper layers produced the strongest results, achieving 95.96% test accuracy.

The experiment demonstrates the potential benefits of transfer learning and fine-tuning for image classification, while also highlighting the importance of validation performance and generalisation.

**The results demonstrate performance on the CIFAKE benchmark only. They should not be interpreted as evidence of reliable detection of AI-generated images from arbitrary real-world sources.**
