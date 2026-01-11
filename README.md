

# Convolutional Neural Networks — Custom Architectures, Transfer Learning, and Data Augmentation

This repository contains my experimental work on **Convolutional Neural Networks (CNNs)** carried out during my **Data Science & Machine Learning (DSML) internship**.
The focus is on **architecture design, representation learning, transferability of features, and generalization behavior**, validated through systematic experimentation on standard vision datasets.

---

## 📌 Objectives

* Understand **hierarchical feature learning** in CNNs
* Analyze the impact of **architectural design choices**
* Study **transfer learning behavior** on small-to-medium datasets
* Evaluate **data augmentation as a regularization strategy**
* Move beyond black-box training toward **controlled experimentation**

---

## 🧠 CNN Architectures Implemented

### Custom CNN Models

* Designed and trained CNNs **from scratch** using Keras
* Implemented modular:

  * Convolution + ReLU blocks
  * Pooling layers for spatial downsampling
  * Fully connected classifiers
* Studied the effect of:

  * Kernel size, stride, and padding on receptive fields
  * Network depth vs overfitting
  * Pooling and dropout as implicit regularization mechanisms

---

## 🔁 Transfer Learning

* Implemented **AlexNet** and **VGG16** architectures using pretrained convolutional backbones
* Evaluated:

  * Feature extraction vs partial fine-tuning strategies
  * Transferability of learned representations across datasets
  * Computational and generalization trade-offs relative to custom CNNs

---

## 🧪 Data Augmentation & Generalization

Data augmentation was treated as **data-space regularization** rather than a heuristic.

Applied transformations include:

* Rotation and scaling
* Horizontal flipping
* Noise injection
* Illumination variation

Experiments analyzed augmentation impact on:

* Overfitting reduction
* Robustness to spatial and appearance variations
* Stability of validation performance

---

## 📊 Datasets Used

* **MNIST**
* **Fashion-MNIST**
* **CIFAR-10**

These datasets enabled comparative analysis across:

* Grayscale vs RGB inputs
* Increasing spatial and semantic complexity
* Small vs moderately complex visual distributions

---

## 🔬 Experimental Focus

* Convergence behavior across datasets
* Representation quality at different network depths
* Performance comparison:

  * Training from scratch vs transfer learning
  * With and without data augmentation
* Generalization trends beyond raw accuracy metrics

---

## ⚙️ Tech Stack

* Python
* TensorFlow
* Keras
* NumPy
* Matplotlib

---

## 📚 Learning Outcomes

This work strengthened my understanding of:

* How **architectural inductive bias** influences feature learning
* Why transfer learning succeeds or fails depending on dataset complexity
* The role of **augmentation-driven invariances** in CNN generalization
* Practical trade-offs between depth, computation, and performance

---

## 📈 Future Work

* Extend experiments to deeper architectures
* Explore batch normalization and modern regularization techniques
* Apply CNNs to higher-resolution and real-world datasets
* Transition from controlled experiments to end-to-end vision tasks

---

