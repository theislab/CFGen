import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

def train_mlp_classifier(adata, covariate, hidden_layer_sizes=(64, 32), epochs=100, batch_size=32, lr=0.001, device='cpu'):
    """
    Train an MLP classifier on a covariate from an AnnData object.

    Parameters:
    - adata: AnnData object
    - covariate: str, the name of the column in adata.obs to classify
    - hidden_layer_sizes: tuple of int, sizes of hidden layers
    - epochs: int, number of training epochs
    - batch_size: int, batch size for training
    - lr: float, learning rate for the optimizer
    - device: str, 'cpu' or 'cuda' for training on GPU/CPU
    
    Returns:
    - model: Trained MLP model
    - class_probs: numpy array, probabilities for each class for each observation
    """
    # Ensure the device is available
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Extract input data (X) and target labels (y)
    X = adata.X if isinstance(adata.X, np.ndarray) else adata.X.toarray()
    y = adata.obs[covariate].values
    
    # Encode categorical labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(le.classes_)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    # Convert to PyTorch tensors
    X_train, X_test = map(torch.FloatTensor, (X_train, X_test))
    y_train, y_test = map(torch.LongTensor, (y_train, y_test))
    
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Define the MLP model
    class MLP(nn.Module):
        def __init__(self, input_dim, hidden_layer_sizes, output_dim):
            super(MLP, self).__init__()
            layers = []
            prev_dim = input_dim
            for hidden_size in hidden_layer_sizes:
                layers.append(nn.Linear(prev_dim, hidden_size))
                layers.append(nn.ReLU())
                prev_dim = hidden_size
            layers.append(nn.Linear(prev_dim, output_dim))
            self.network = nn.Sequential(*layers)
        
        def forward(self, x):
            return self.network(x)
    
    # Instantiate the model
    input_dim = X.shape[1]
    model = MLP(input_dim, hidden_layer_sizes, num_classes).to(device)
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    
            # Evaluate on the test set
            model.eval()
            with torch.no_grad():
                X_test = X_test.to(device)
                y_test = y_test.to(device)
                test_logits = model(X_test)
                test_predictions = torch.argmax(test_logits, dim=1)  # Predicted class
                test_accuracy = (test_predictions == y_test).float().mean().item()  # Accuracy
        
            print(f"Test Accuracy: {test_accuracy:.4f}")
    
    print("Training complete.")
    return model, le
