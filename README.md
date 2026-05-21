# 🔥 Fire Detection Pipeline using Deep Learning

## 📌 Project Overview

This project focuses on **Fire Detection using Deep Learning and Computer Vision** techniques.
The objective is to classify images into two categories:

* 🔥 **Fire**
* 🌲 **Non-Fire**

Several Convolutional Neural Network (CNN) architectures were implemented and compared to evaluate their performance on a fire image dataset.

---

# 📂 Dataset

Dataset used: **phylake1337/fire-dataset**

## 📊 Dataset Statistics

| Category    | Number of Images |
| ----------- | ---------------- |
| 🔥 Fire     | 755              |
| 🌲 Non-Fire | 244              |
| 📦 Total    | 999              |

## 🔀 Dataset Split

| Set        | Percentage |
| ---------- | ---------- |
| Train      | 70%        |
| Validation | 10%        |
| Test       | 20%        |

---

# 🛠️ Technologies Used

* Python
* PyTorch
* torchvision
* NumPy
* Matplotlib
* scikit-learn
* Streamlit

---

# 🔄 Deep Learning Pipeline

## 1️⃣ Data Preprocessing

The following preprocessing steps were applied:

* Resize images to **224 × 224**
* Convert images to tensors
* Normalize using ImageNet statistics

---

## 2️⃣ Data Augmentation

To improve model generalization and reduce overfitting, several augmentation techniques were used:

* Horizontal Flip
* Vertical Flip
* Random Rotation
* ColorJitter
* Gaussian Blur

---

# 🧠 Model Architectures

Three different CNN architectures were tested and compared.

| Model               | Description                                 |
| ------------------- | ------------------------------------------- |
| **SimpleCNN**       | Custom CNN built from scratch               |
| **ResNet-50**       | Transfer Learning using pretrained ResNet50 |
| **EfficientNet-B0** | Fine-tuned EfficientNet model               |

---

# 🔹 Model Details

## 🧱 SimpleCNN

Custom CNN architecture containing:

* 4 Convolutional Blocks
* Batch Normalization
* ReLU Activation
* MaxPooling
* Dropout Regularization

### Training Configuration

* Optimizer: Adam
* Learning Rate: `3e-4`

---

## 🔹 ResNet-50

Transfer Learning approach using a pretrained ResNet50 model.

### Configuration

* Pretrained on ImageNet
* Frozen backbone layers
* Custom fully connected classifier
* Dropout: `0.3`

### Training Configuration

* Optimizer: Adam
* Learning Rate: `1e-4`

---

## 🔹 EfficientNet-B0

EfficientNet-B0 achieved the best overall performance.

### Configuration

* Full fine-tuning
* AdamW optimizer
* Weight decay regularization
* CosineAnnealingLR scheduler

---

# 📊 Results

| Model           | Validation Accuracy |
| --------------- | ------------------- |
| SimpleCNN       | 96.5%               |
| ResNet-50       | 76.5%               |
| EfficientNet-B0 | **99.5% 🏆**        |

---

# 🏆 Best Model: EfficientNet-B0

EfficientNet-B0 achieved:

* ✅ **99.5% Validation Accuracy**
* ✅ Excellent fire/non-fire classification
* ✅ Strong generalization capability
* ✅ Stable training performance

---

# 📈 Evaluation Metrics

The models were evaluated using:

* Accuracy
* Validation Loss
* ROC Curve
* Confusion Matrix
* Classification Report
* Grad-CAM Visualization

---

# 👁️ Grad-CAM Visualization

Grad-CAM was used to visualize the regions influencing model predictions.

The model mainly focused on:

* 🔥 Flame regions
* 🌫️ Smoke areas
* 🌲 Background context

This improves the explainability and reliability of the model predictions.

---

# ⚠️ Challenges

Several challenges were encountered during the project:

* Limited dataset size
* Class imbalance
* Visual similarity between some fire and non-fire images
* Risk of overfitting

---

# 🚀 Future Improvements

Possible future enhancements include:

* Mixup augmentation
* WeightedRandomSampler
* Progressive fine-tuning
* Model ensembling
* Real-time fire detection from video streams
* Deployment using Streamlit or Flask
* Edge-device optimization

---

# 📁 Project Structure

```bash
FIRE-DETECTION-CNN
├───anaconda_projects
│   └───db
├───app
│   ├───sample_images
│   └───__pycache__
├───dataset
│   └───fire_dataset
│       ├───fire_images
│       └───non_fire_images
├───notebooks
│   ├───drafts
│   └───final
└───presentation
```

---

# 🚀 Installation

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/mohamedhamed-ibnhadjmohamed/fire-detection-cnn.git
cd fire-detection-cnn
```

## 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run Training

```bash
python train.py
```

---

# ▶️ Launch Streamlit Application

```bash
cd app
streamlit run streamlit_app.py
```

---

# 📷 Example Predictions

| Image Type   | Prediction  |
| ------------ | ----------- |
| Fire Image   | 🔥 Fire     |
| Forest Image | 🌲 Non-Fire |

---

# 📊 Models Comparison

| Model                  | Accuracy  | AUC Score | Parameters |
| ---------------------- | --------- | --------- | ---------- |
| CNN Simple             | 96.5%     | —         | —          |
| ResNet50 (Fine-Tuning) | 76.5%     | —         | —          |
| EfficientNet-B0        | **99.5%** | —         | —          |

---

# 📚 Conclusion

This project demonstrates the effectiveness of **Deep Learning** and **Transfer Learning** for image classification tasks related to fire detection.

Among all tested architectures, **EfficientNet-B0** achieved the best performance and proved highly effective for detecting fire in images with excellent accuracy and generalization.

The project can be extended into a real-time fire monitoring system for smart surveillance and safety applications.

---

# 👨‍💻 Author

Developed as a **Deep Learning and Computer Vision** project using **PyTorch** and **Streamlit**.
