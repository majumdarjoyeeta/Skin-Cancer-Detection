# Skin-Cancer-Detection
# 🌟 Skin Cancer Detection Project 🌟

Welcome to the **Skin Cancer Detection** project! This project utilizes deep learning techniques to classify skin cancer images using the **DenseNet121** architecture. Below, you'll find detailed information about the dataset, model architecture, training parameters, results, and more.

## 📥 GitHub Repository

You can find the complete code and resources for this project on GitHub: [Skin Cancer Detection Repository](https://github.com/majumdarjoyeeta/Skin-Cancer-Detection)

---

## 📊 Dataset Information

This dataset is used to train a deep learning model for **binary classification tasks**. The model is based on the **DenseNet121 architecture**, which is a type of convolutional neural network (CNN) that is well-suited for image classification tasks.

---

## 🏗️ Model Architecture

The model consists of the following layers:

- **Input Layer**: 224x224x3 images
- **DenseNet121 Base Model**: Pre-trained on ImageNet weights, without the top (classification) layers
- **Global Average Pooling Layer**
- **Fully Connected Layer**: 1024 units with ReLU activation
- **Batch Normalization Layer**
- **Dropout Layer**: 0.8 dropout rate
- **Fully Connected Layer**: 512 units with ReLU activation
- **Batch Normalization Layer**
- **Dropout Layer**: 0.5 dropout rate
- **Final Layer**: 1 unit with sigmoid activation

---

## ⚙️ Training Parameters

- **Optimizer**: Adam
- **Loss Function**: Binary cross-entropy
- **Metrics**: Accuracy
- **Batch Size**: 32
- **Number of Epochs**: 150
- **Validation Split**: 0.2

---

## 🏆 Training Results

The model was trained on a dataset of **132 images**, with a validation split of **0.2**. The training process took approximately **10 hours** to complete. The model achieved a **validation accuracy of 88%**, which is higher than the initial test result.

---

## 📈 Model Evaluation

The model was evaluated on a test dataset of **132 images** and achieved an accuracy of **88%**. The model's performance can be improved with **hyperparameter tuning** and **data augmentation techniques**.

---

## 🚀 Model Deployment

The trained model can be deployed in a variety of applications, including:

- **Image Classification**
- **Object Detection**
- **Image Segmentation**

The model can be utilized across various industries, including **healthcare**, **finance**, and **retail**.

---

## 💻 Code

The code for the model is written in **Python**, using the **Keras** deep learning library. The code is well-documented and easy to follow.

---

## 📋 Requirements

To run the code, ensure you have the following packages installed:

- **Python**: 3.6+
- **Keras**: 2.3.1+
- **TensorFlow**: 2.3.1+
- **NumPy**: 1.19.5+
- **SciPy**: 1.5.4+
- **Matplotlib**: 3.3.4+
- **Scikit-learn**: 0.24.1+

---

## 🎉 Conclusion

This README provides a comprehensive overview of the **Skin Cancer Detection** project. With the right tuning and deployment, this model can be a powerful tool in various applications! 🌟

---

