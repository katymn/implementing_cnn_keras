# Implementing a Convolutional Neural Network Using Keras

This project demonstrates how to build and train a Convolutional Neural Network (CNN) to classify handwritten digits from the MNIST dataset using TensorFlow Keras.

---

## 📖 Overview

Handwritten digit recognition is a classic computer vision problem widely used to benchmark machine learning models. The MNIST dataset contains 70,000 grayscale images of handwritten digits (0-9) that serve as a standard dataset for image classification tasks.

In this lab, we implement a CNN from scratch to classify these digits, following these key steps:

- Define the machine learning problem
- Import and explore the dataset
- Preprocess and prepare data for modeling
- Construct a CNN architecture using Keras
- Train the CNN model
- Evaluate the model’s performance on test data
- Visualize predictions

---

## 🧩 Project Structure

- Load and inspect MNIST dataset (60,000 training, 10,000 test images)
- Normalize pixel values to the range [0, 1]
- Reshape input data to include a single color channel (28x28x1)
- Build a CNN with four convolutional layers, batch normalization, ReLU activation, and a global average pooling layer
- Compile the model using stochastic gradient descent (SGD) optimizer and sparse categorical cross-entropy loss
- Train the model for one epoch
- Evaluate the model on test data and visualize predictions

---

## ⚙️ Technologies Used

- Python 3.x
- TensorFlow / Keras
- NumPy
- Matplotlib
- Seaborn

---

## 🔍 How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/heat-illness-diagnostic.git
   
2. Navigate to the project directory:
   ```bash
   cd heat-illness-diagnostic

3. Install required packages (ideally in a virtual environment):
   ```bash
   pip install tensorflow numpy matplotlib seaborn
   
4. Open the Jupyter notebook and run all cells, or run the script directly if converted.

---

## 📈 Model Summary
- Input: 28x28 grayscale images with 1 channel
- Architecture:
  - 4 Conv2D layers with increasing filters: 16, 32, 64, 128
  - Batch Normalization and ReLU activation after each conv layer
  - Global Average Pooling layer
  - Dense output layer with 10 units (digits 0-9)
- Optimizer: SGD with learning rate 0.1
- Loss Function: Sparse Categorical Crossentropy (from logits)
- Epochs: 1 (for demonstration purposes)

---

## 📊 Results
- Training Accuracy: ~91.6%
- Test Accuracy: ~92.8%
- Loss and accuracy are printed after training and evaluation
- Visualizations display sample test images with predicted labels

---

## 📝 Notes
Training for only 1 epoch is sufficient for this demonstration but can be extended for improved accuracy.

The notebook uses GPU acceleration if available for faster training.

The model uses batch normalization and global average pooling to improve training stability and performance.4

---

## 🙋‍♀️ Contact
@katymn
Software Engineering Student | Machine Learning Enthusiast

Feel free to reach out for questions or collaboration!

---

## 📄 Credit
Break Through Tech with Cornell Tech
