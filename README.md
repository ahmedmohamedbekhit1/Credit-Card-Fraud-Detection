
# Credit Card Fraud Detection using Neural Networks

This project implements a deep learning model to detect fraudulent credit card transactions. It uses PyTorch to train a simple feedforward neural network for binary classification (fraud or non-fraud). The dataset is highly imbalanced, and techniques like SMOTE (Synthetic Minority Over-sampling Technique) are used to address this imbalance.

## Table of Contents

1. [Installation](#installation)
2. [Dataset](#dataset)
3. [Usage](#usage)
4. [Code Walkthrough](#code-walkthrough)
5. [Results](#results)
6. [License](#license)

---

## Installation

### Dependencies

To run this code, install the following Python libraries:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `torch` (PyTorch)
- `scikit-learn`
- `imblearn`

You can install them using `pip`:

```bash
pip install pandas numpy matplotlib seaborn torch scikit-learn imbalanced-learn
```

If you're using **GPU** for faster training, ensure that you have the appropriate CUDA version installed for PyTorch.

---

## Dataset

This project uses the **Credit Card Fraud Detection dataset**, which can be found on Kaggle. The dataset contains 31 features and a target column `Class`, where:
- `0` represents non-fraudulent transactions.
- `1` represents fraudulent transactions.

You can download the dataset from Kaggle at:  
[https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

Place the `creditcard.csv` file in the project directory before running the code.

---

## Usage

### Running the Code

1. Install the dependencies.
2. Download the dataset from Kaggle and place the `creditcard.csv` file in the project directory.
3. Run the script:

```bash
python fraud_detection.py
```

The script will perform the following steps:
- Train a neural network to classify fraudulent transactions.
- Display various performance metrics and visualizations.

---

## Code Walkthrough

### Step 1: Load the Dataset

The dataset is loaded using `pandas`, and basic exploratory data analysis is done to understand the distribution of fraudulent and non-fraudulent transactions.

### Step 2: Visualize Class Distribution

A `countplot` is generated to visualize the class distribution of fraudulent and non-fraudulent transactions. Given the class imbalance, this plot will highlight the disproportion between the two classes.

### Step 3: Data Preprocessing

- The features (`X`) and target (`y`) are separated.
- The features are scaled using `StandardScaler` to normalize the data.
- **SMOTE** is applied to balance the class distribution by generating synthetic samples for the minority class (fraudulent transactions).

### Step 4: Build the Neural Network

A simple feedforward neural network is implemented using **PyTorch**, with the following architecture:
- Input layer
- Two hidden layers with **ReLU** activations
- Output layer with **sigmoid** activation for binary classification.

### Step 5: Train the Model

The model is trained using **Binary Cross-Entropy Loss** and the **Adam optimizer** for 20 epochs. The loss is printed for each epoch to monitor the training progress.

### Step 6: Evaluate the Model

After training, the model is evaluated on the test set. The following metrics are calculated:
- **Confusion Matrix**: Shows the true positives, false positives, true negatives, and false negatives.
- **Classification Report**: Precision, recall, F1-score, and support for each class.
- **ROC Curve**: The area under the curve (AUC) and the trade-off between the true positive rate (sensitivity) and the false positive rate (1-specificity).

### Step 7: Visualizations

The following visualizations are generated:
- **Class Distribution Plot**: Displays the imbalance between fraudulent and non-fraudulent transactions.
- **Training Loss Curve**: Shows how the model's loss decreases during training.
- **Confusion Matrix Heatmap**: Visualizes the true vs predicted class counts.
- **ROC Curve**: Illustrates the model’s performance in distinguishing between classes.

---

## Results

### Class Distribution
Before training, we visualize the class distribution of the dataset. As shown below, there is a clear imbalance between fraudulent and non-fraudulent transactions.

![Class Distribution Plot](./images/class_distribution.png)

### Training Loss Curve
The training loss curve shows the loss function decreasing over epochs, indicating that the model is learning.

![Training Loss Curve](./images/training_loss_curve.png)

### Confusion Matrix
After training the model, we evaluate its performance using the confusion matrix. Here’s the confusion matrix that shows how the model performs on the test set:

```
Confusion Matrix (rows: actual, columns: predicted):
[[15406 41458]
 [    2    96]]
```

![Confusion Matrix Heatmap](./images/confusion_matrix.png)

### ROC Curve
The ROC curve illustrates the model's ability to distinguish between fraudulent and non-fraudulent transactions. The area under the curve (AUC) provides insight into the overall model performance.

![ROC Curve](./images/roc_curve.png)

### Classification Report
The classification report shows the precision, recall, and F1-score for both classes. Due to the class imbalance, the model achieves:
- **High recall** for fraudulent transactions (98%), but a **low precision** (0%).
- **Overall accuracy** of **27.21%**, which is largely influenced by the class imbalance.

```
              precision    recall  f1-score   support

           0       1.00      0.27      0.43     56864
           1       0.00      0.98      0.00        98

    accuracy                           0.27     56962
   macro avg       0.50      0.63      0.22     56962
weighted avg       1.00      0.27      0.43     56962
```

**Accuracy:** 0.2721

This suggests the model is biased towards predicting non-fraudulent transactions, as expected with an imbalanced dataset.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

