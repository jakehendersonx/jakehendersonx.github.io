---
title: "Neural Network Intuition: A Neural Network that Adds Two Numbers in PyTorch"
date: 2020-09-15T11:30:03+00:00
weight: 1
# aliases: ["/first"]
tags: ["neural network", "machine learning", "pytorch"]
author: "Jake Henderson"
# author: ["Me", "You"] # multiple authors
showToc: true
TocOpen: false
draft: false
hidemeta: false
comments: false
description: "writing a nn that adds two numbers in pytorch and comparing with the first response of flagship AI models"
canonicalURL: "https://canonical.url/to/page"
disableHLJS: true # to disable highlightjs
disableShare: true
disableHLJS: false
hideSummary: false
searchHidden: true
ShowReadingTime: true
ShowBreadCrumbs: false
ShowPostNavLinks: false
ShowWordCount: true
ShowRssButtonInSectionTermList: false
UseHugoToc: true
cover:
    image: "<image path/url>" # image path/url
    alt: "<alt text>" # alt text
    caption: "<text>" # display caption under cover
    relative: false # when using page bundles set this to true
    hidden: true # only hide on current single page
editPost:
    URL: "https://github.com/<path_to_repo>/content"
    Text: "Suggest Changes" # edit text
    appendFilePath: true # to append file path to Edit link
---

## Outline
- brief intro of why do problems like this to test intuition
- link to project code 
- a neural net approximates this equation
- simple addition of 2 numers approximates a simpler equation
    that is now just about converging the x weights and error 
- what does normalization do
- go over MSE loss and Adam Optimizer -> link to resources
- generate the data and train the neural networks (mine, claude, and )

### What Function are we Approximating

### Defining AddNet


```python
import torch
import torch.nn as nn
import numpy as np


class AddNet(nn.Module):
    def __init__(self):
        super(AddNet, self).__init__()
        #TODO: explain simple linear equation facilitated
        self.output = nn.Linear(2, 1)
    
    def forward(self, x):
        return self.output(x)
```

### Normalization and Data Generation
```python
def normalize(x):
    """Normalizes input values to range [0,1]."""
    return x / 100.0

def denormalize(x):
    """Denormalizes values back to original range."""
    return x * 100.0

def generate_training_data(num_samples):
    x1 = torch.randint(0, 101, (num_samples, 1), dtype=torch.float32)
    x2 = torch.randint(0, 101, (num_samples, 1), dtype=torch.float32)
    inputs = torch.cat((normalize(x1), normalize(x2)), dim=1)
    targets = normalize(x1 + x2)
    return inputs, targets
```
### Loss Function, Optimizer, and 
```python
def train_add_net(model, num_epochs=500, batch_size=32):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    
    inputs, targets = generate_training_data(1000)
    
    for epoch in range(num_epochs):
        idx = torch.randperm(inputs.shape[0])[:batch_size]
        batch_inputs = inputs[idx]
        batch_targets = targets[idx]
        
        outputs = model(batch_inputs)
        loss = criterion(outputs, batch_targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
            print_weights_and_bias(model)
```

### Inference

```python
def inference(model, x1, x2):
    model.eval()
    with torch.no_grad():
        x1 = max(0, min(100, int(x1)))
        x2 = max(0, min(100, int(x2)))
        
        x1_norm = normalize(torch.tensor([[float(x1)]], dtype=torch.float32))
        x2_norm = normalize(torch.tensor([[float(x2)]], dtype=torch.float32))
        inputs = torch.cat((x1_norm, x2_norm), dim=1)
        predicted_sum_norm = model(inputs)
        return denormalize(predicted_sum_norm).item()

```

### Results

```python
def batch_test(model, num_samples):
    print(f"\nBatch Testing ({num_samples} samples):")
    print("-" * 50)
    
    total_percent_error = 0
    
    for _ in range(num_samples):
        x1 = np.random.randint(0, 101)
        x2 = np.random.randint(0, 101)
        predicted_sum = test_addition(model, x1, x2)
        actual_sum = x1 + x2
        percent_error = abs(predicted_sum - actual_sum) / actual_sum * 100
        total_percent_error += percent_error
    
    average_percent_error = total_percent_error / num_samples
    print(f"\nAverage Percent Error across {num_samples} samples: {average_percent_error:.2f}%")
```

### Flagship AI Model's Responce to the prompt
```
Write a neural network in PyTorch that can add two numbers. 
Call it AddNet. 
It should add two whole numbers that are between 0 and 100.
Goal 1 is to get it to add two arbitrary whole numbers in a defined range with the lowest possible error. Goal 2 is to keep the neural net and training definitions to be as simple as possible while still accomplishing Goal 1.
At each relevant step of your network definition and training, write about why you made the choice to use that "technique".
```

###### ChatGPT-4o's Atttempt
```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define the range of input numbers
NUM_RANGE = 101  # From 0 to 100 inclusive

# Define the AddNet model
class AddNet(nn.Module):
    def __init__(self):
        super(AddNet, self).__init__()
        # Simple feedforward network with 1 hidden layer
        self.fc1 = nn.Linear(2, 10)  # 2 inputs (two numbers) -> 10 hidden neurons
        self.fc2 = nn.Linear(10, 1)  # 10 hidden neurons -> 1 output (sum)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))  # ReLU activation for non-linearity
        x = self.fc2(x)  # No activation for output layer (regression task)
        return x

# Step 1: Initialize the model, loss function, and optimizer
model = AddNet()
criterion = nn.MSELoss()  # Mean Squared Error Loss for regression
optimizer = optim.Adam(model.parameters(), lr=0.01)  # Adam optimizer for fast convergence

# Explain choice of architecture:
# - Input: Two numbers as a 2D tensor (e.g., [45, 30])
# - Hidden layer: A small fully connected layer with 10 neurons balances simplicity and capacity
# - ReLU: Adds non-linearity to allow the network to learn complex relationships
# - Output: Single value (sum), no activation because this is a regression problem

# Step 2: Generate training data
def generate_data(size):
    x_data = np.random.randint(0, NUM_RANGE, size=(size, 2))  # Random pairs of integers
    y_data = np.sum(x_data, axis=1, keepdims=True)  # Compute sums
    return torch.tensor(x_data, dtype=torch.float32), torch.tensor(y_data, dtype=torch.float32)

train_x, train_y = generate_data(10000)  # Generate 10,000 training examples

# Step 3: Train the model
epochs = 500
for epoch in range(epochs):
    optimizer.zero_grad()  # Clear gradients
    predictions = model(train_x)  # Forward pass
    loss = criterion(predictions, train_y)  # Compute loss
    loss.backward()  # Backward pass
    optimizer.step()  # Update weights

    # Print progress
    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Explain training process:
# - Loss: MSE is suitable for regression because it penalizes large errors
# - Optimizer: Adam is chosen for its adaptive learning rate and robustness
# - Epochs: 500 iterations are enough to converge for this simple task
# - Data: 10,000 examples provide sufficient coverage of input space (0 to 100)

# Step 4: Test the model
test_x, test_y = generate_data(1000)  # Generate test data
with torch.no_grad():  # No gradient computation during testing
    test_predictions = model(test_x)
    test_loss = criterion(test_predictions, test_y)
    print(f"Test Loss: {test_loss.item():.4f}")

# Step 5: Demonstrate prediction
example = torch.tensor([[23, 45]], dtype=torch.float32)
with torch.no_grad():
    prediction = model(example).item()
    print(f"Prediction for input {example.tolist()}: {prediction:.2f} (Expected: {sum(example.tolist()[0])})")

```
###### Claude 3.5 Sonnet's Attempt
```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class AddNet(nn.Module):
    def __init__(self):
        super(AddNet, self).__init__()
        # A simple architecture with two hidden layers
        self.network = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 1)
        )
    
    def forward(self, x):
        return self.network(x)

# Initialize the model
model = AddNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Generate training data
def generate_data(num_samples=1000):
    # Generate random numbers between -10 and 10
    x1 = np.random.uniform(-10, 10, (num_samples, 1))
    x2 = np.random.uniform(-10, 10, (num_samples, 1))
    # True sums
    y = x1 + x2
    # Convert to PyTorch tensors
    X = torch.FloatTensor(np.hstack((x1, x2)))
    y = torch.FloatTensor(y)
    return X, y

# Training loop
def train(epochs=1000):
    for epoch in range(epochs):
        X, y = generate_data()
        
        # Forward pass
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}')

# Train the model
train()

# Test the model
def test_addition(x1, x2):
    model.eval()
    with torch.no_grad():
        input_tensor = torch.FloatTensor([[x1, x2]])
        prediction = model(input_tensor)
        return prediction.item()

# Example usage
x1, x2 = 5.7, 3.2
predicted_sum = test_addition(x1, x2)
actual_sum = x1 + x2
print(f"\nTest Result:")
print(f"Numbers: {x1} + {x2}")
print(f"Predicted sum: {predicted_sum:.4f}")
print(f"Actual sum: {actual_sum:.4f}")
print(f"Error: {abs(predicted_sum - actual_sum):.4f}")
```
