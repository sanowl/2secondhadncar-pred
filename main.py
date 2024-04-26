import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the data
data = pd.read_csv('cars.csv')

# Preprocess the data
categorical_cols = ['Brand', 'Fuel_Type', 'Transmission', 'Owner_Type']
numerical_cols = ['Year', 'Kilometers_Driven', 'Mileage', 'Engine', 'Power', 'Seats']

for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])

X = data[categorical_cols + numerical_cols]
y = data['Price']

# Split data into train, validation, and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Normalize numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Convert data to PyTorch tensors
X_train = torch.from_numpy(X_train.astype('float32'))
X_val = torch.from_numpy(X_val.astype('float32'))
X_test = torch.from_numpy(X_test.astype('float32'))
y_train = torch.from_numpy(y_train.values.astype('float32')).unsqueeze(1)
y_val = torch.from_numpy(y_val.values.astype('float32')).unsqueeze(1)
y_test = torch.from_numpy(y_test.values.astype('float32')).unsqueeze(1)

# Define the model
class CarPriceModel(nn.Module):
    def __init__(self, input_size):
        super(CarPriceModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.relu(self.fc3(out))
        out = self.dropout(out)
        out = self.fc4(out)
        return out

input_size = X_train.shape[1]
model = CarPriceModel(input_size)

# Define the loss function and optimizer
criterion = nn.HuberLoss()  # Using Huber loss for robustness
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # Smaller learning rate

# Train the model
num_epochs = 500
for epoch in range(num_epochs):
    model.train()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Evaluate on validation set
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val)
    
    if (epoch+1) % 50 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

# Evaluate the model on test set
model.eval()
with torch.no_grad():
    predictions = model(X_test)
    mae = nn.L1Loss()(predictions, y_test)
    print(f'Mean Absolute Error: {mae.item()}')