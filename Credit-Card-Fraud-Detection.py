import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc
from imblearn.over_sampling import SMOTE
from torch import nn, optim

# Step 1: Load the dataset
print("\n=== Step 1: Loading the Dataset ===")
df = pd.read_csv(r"D:\creditcard.csv")
print(f"Total transactions: {len(df)}")
print(f"Sample data:\n{df.head()}")

# Step 2: Identify and display fraudulent transactions
fraud_cases = df[df['Class'] == 1]
non_fraud_cases = df[df['Class'] == 0]
print(f"\nFraudulent transactions: {len(fraud_cases)}")
print(f"Non-fraudulent transactions: {len(non_fraud_cases)}")

# Visualize class distribution
# Visualize class distribution with counts and percentages
plt.figure(figsize=(8, 6))
sns.countplot(x='Class', data=df)
plt.title('Class Distribution (0: Non-Fraud, 1: Fraud)')
plt.xlabel('Class')
plt.ylabel('Count')

# Annotate the plot with percentages
total = len(df)
for p in plt.gca().patches:
    count = p.get_height()
    percentage = f"{(count / total) * 100:.2f}%"
    plt.gca().annotate(f'{count} ({percentage})',
                       xy=(p.get_x() + p.get_width() / 2, count),
                       ha='center', va='bottom')

plt.show()

# Step 3: Prepare the data
print("\n=== Step 3: Preparing the Data ===")
X = df.drop('Class', axis=1).values
y = df['Class'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Handle imbalance with SMOTE
print("Applying SMOTE to balance the data...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
print(f"Resampled dataset size: {len(X_train_resampled)}")

# Step 4: Define the model
print("\n=== Step 4: Defining the Neural Network ===")
class FraudDetectionNN(nn.Module):
    def __init__(self):
        super(FraudDetectionNN, self).__init__()
        self.fc1 = nn.Linear(X_train_resampled.shape[1], 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = FraudDetectionNN().to(device)

# Step 5: Train the model
print("\n=== Step 5: Training the Model ===")
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

X_train_tensor = torch.tensor(X_train_resampled, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train_resampled, dtype=torch.float32).unsqueeze(1).to(device)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)

losses = []
epochs = 20
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss.item():.4f}")

# Step 6: Plot training loss
plt.figure(figsize=(8, 6))
plt.plot(range(1, epochs + 1), losses, marker='o')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()

# Step 7: Evaluate the model
print("\n=== Step 7: Evaluating the Model ===")
model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor).cpu().numpy()
    y_pred_class = (y_pred > 0.5).astype(int)

# Step 8: Display confusion matrix and heatmap
conf_matrix = confusion_matrix(y_test, y_pred_class)
print("Confusion Matrix (rows: actual, columns: predicted):")
print(conf_matrix)

sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix Heatmap')
plt.xlabel('Predicted Class')
plt.ylabel('Actual Class')
plt.show()

# Step 9: Plot ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# Step 10: Print classification report
print("\n=== Step 10: Classification Report ===")
print(classification_report(y_test, y_pred_class))
accuracy = accuracy_score(y_test, y_pred_class)
print(f"\nAccuracy: {accuracy:.4f}")
