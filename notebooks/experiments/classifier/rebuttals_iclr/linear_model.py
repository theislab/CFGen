import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm 

class MultiClassLogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MultiClassLogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)  # Output layer with num_classes

    def forward(self, x):
        return self.linear(x)

def train_and_predict_multiclass_logistic_regression_torch(X_train, X_test, y_train, y_test, batch_size=64, epochs=10, learning_rate=0.001):
    # Convert data into a tensor 
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long) 
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # Create DataLoader for batch processing
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize the model, loss function, and optimizer
    input_dim = X_train.shape[1]  # Number of features
    num_classes = len(torch.unique(y_train_tensor))  # Number of unique classes
    model = MultiClassLogisticRegressionModel(input_dim, num_classes)
    criterion = nn.CrossEntropyLoss()  # CrossEntropyLoss includes softmax and negative log-likelihood
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    for epoch in tqdm(range(epochs)):
        model.train()  # Set model to training mode
        running_loss = 0.0

        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            optimizer.zero_grad()  # Zero gradients
            y_pred = model(X_batch)  # Forward pass
            loss = criterion(y_pred, y_batch)  # Compute the loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update the weights

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}")

    # Switch model to evaluation mode
    model.eval()

    # Make predictions on the test set
    with torch.no_grad():
        y_pred_prob = model(X_test_tensor)  # Get raw predictions
        y_pred = torch.argmax(y_pred_prob, dim=1)  # Get class with max probability

    # Calculate the accuracy score
    accuracy = accuracy_score(y_test_tensor, y_pred)

    # Generate a classification report
    report = classification_report(y_test_tensor, y_pred, output_dict=True)

    # Return the model, accuracy, and report
    return model, accuracy, report
