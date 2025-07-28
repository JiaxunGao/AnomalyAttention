import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, f1_score

from model_implementation import GazeTransformer
from data_processing import load_data, prepare_fold_data, create_data_loaders
from utils import generate_random_mask


def train_epoch(model, train_loader, val_loader, criterion, optimizer, device, num_patches, k=0):
    """
    Train the model for one epoch.
    
    Args:
        model: The GazeTransformer model
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Computing device
        num_patches: Number of patches for masking
        k: Starting index for sequence slicing
    
    Returns:
        train_accuracy: Training accuracy for the epoch
        val_accuracy: Validation accuracy
        val_loss: Validation loss
    """
    total = 0
    total_correct = 0
    
    for i, (eye, label) in enumerate(train_loader):
        eye = eye[:, k:, :]  # Slice sequence if needed
        model.train()
        eye = eye.to(device)
        label = label.to(device)
        
        # Generate random mask
        mask = generate_random_mask(eye.size(0), num_patches)
        mask = mask.to(device)
        
        # Forward pass
        output, attn_weights = model(eye, mask=mask)
        loss = criterion(output, label)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        _, predicted = torch.max(output, 1)
        correct = (predicted == label).sum().item()
        total_correct += correct
        total += label.size(0)

        # Print progress every 5 iterations
        if i % 5 == 0:
            accuracy = 100 * correct / label.size(0)
            print(f"Iteration {i}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%")
            
            val_accuracy, val_loss = evaluate(model, val_loader, criterion, device, num_patches, k)
            print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

            # Early stopping condition
            if val_accuracy > 89:
                break

    train_accuracy = 100 * total_correct / total
    return train_accuracy, val_accuracy, val_loss


def evaluate(model, test_loader, criterion, device, num_patches, k=0):
    """
    Evaluate the model on test data.
    
    Args:
        model: The GazeTransformer model
        test_loader: Test data loader
        criterion: Loss function
        device: Computing device
        num_patches: Number of patches for masking
        k: Starting index for sequence slicing
    
    Returns:
        accuracy: Test accuracy
        avg_loss: Average test loss
    """
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    
    with torch.no_grad():
        for eye, label in test_loader:
            eye = eye[:, k:, :]  # Slice sequence if needed
            eye = eye.to(device)
            label = label.to(device)
            
            # Generate random mask
            mask = generate_random_mask(eye.size(0), num_patches)
            mask = mask.to(device)
            
            # Forward pass
            output, attn_weights = model(eye, mask=mask)
            
            # Calculate predictions and accuracy
            _, predicted = torch.max(output, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
            
            # Calculate loss
            loss = criterion(output, label)
            total_loss += loss.item()

    accuracy = 100 * correct / total
    avg_loss = total_loss / len(test_loader)
    return accuracy, avg_loss


def cross_validation_training(eye, label, model_params, training_params):
    """
    Perform k-fold cross-validation training.
    
    Args:
        eye: Complete eye movement data
        label: Complete labels
        model_params: Dictionary containing model parameters
        training_params: Dictionary containing training parameters
    
    Returns:
        fold_accuracy: List of validation accuracies for each fold
        fold_train_accuracy: List of training accuracies for each fold
    """
    # Extract parameters
    circle_length = model_params['circle_length']
    num_patches = model_params['num_patches']
    embedding_dim = model_params['embedding_dim']
    num_heads = model_params['num_heads']
    num_layers = model_params['num_layers']
    num_classes = model_params['num_classes']
    
    epochs = training_params['epochs']
    k_fold = training_params['k_fold']
    learning_rate = training_params.get('learning_rate', 1e-4)
    weight_decay = training_params.get('weight_decay', 1e-2)
    device = training_params['device']
    k = training_params.get('k', 0)  # Sequence slicing parameter
    
    # Initialize cross-validation
    kf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=42)
    fold_accuracy = []
    fold_train_accuracy = []

    for fold, (train_index, test_index) in enumerate(kf.split(eye, label)):
        print(f"\n{'='*50}")
        print(f"Fold {fold + 1}/{k_fold}")
        print(f"{'='*50}")
        
        # Initialize model, criterion, optimizer, and scheduler for each fold
        model = GazeTransformer(
            circle_length, num_patches, embedding_dim, 
            num_heads, num_layers, num_classes
        ).to(device)
        
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=5, cooldown=0.5, min_lr=5e-6, verbose=True
        )
        
        # Prepare fold data
        train_eye, train_label, test_eye, test_label = prepare_fold_data(
            eye, label, train_index, test_index, balance_data=True
        )
        
        # Create data loaders
        train_loader, val_loader = create_data_loaders(
            train_eye, train_label, test_eye, test_label,
            train_batch_size=8, val_batch_size=32
        )
        
        best_val_accuracy = 0
        
        # Training loop
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 30)
            
            train_accuracy, val_accuracy, val_loss = train_epoch(
                model, train_loader, val_loader, criterion, 
                optimizer, device, num_patches, k
            )
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Early stopping condition
            if val_accuracy > 90:
                print("Early stopping triggered!")
                break
            
            # Track best validation accuracy
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                # Save best model
                torch.save(model.state_dict(), f"best_model_fold_{fold}.pth")

        # Store fold results
        fold_accuracy.append(best_val_accuracy)
        fold_train_accuracy.append(train_accuracy)
        
        print(f"\nFold {fold + 1} Results:")
        print(f"Best Validation Accuracy: {best_val_accuracy:.2f}%")
        print(f"Final Training Accuracy: {train_accuracy:.2f}%")

    # Save final model
    torch.save(model.state_dict(), "PD_Gaze_Cls.pth")
    
    return fold_accuracy, fold_train_accuracy


def main():
    """
    Main training function.
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    eye, label = load_data('train_transformer.pkl', 'test_transformer.pkl')
    
    # Model parameters
    model_params = {
        'circle_length': 80,
        'num_patches': 1520 // 80,  # sequence_length // circle_length
        'embedding_dim': 128,
        'num_heads': 4,
        'num_layers': 3,
        'num_classes': 2
    }
    
    # Training parameters
    training_params = {
        'epochs': 70,
        'k_fold': 5,
        'learning_rate': 1e-4,
        'weight_decay': 1e-2,
        'device': device,
        'k': 0  # Starting index for sequence slicing
    }
    
    # Perform cross-validation training
    fold_accuracy, fold_train_accuracy = cross_validation_training(
        eye, label, model_params, training_params
    )
    
    # Print final results
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Average Validation Accuracy: {np.mean(fold_accuracy):.2f}% ± {np.std(fold_accuracy):.2f}%")
    print(f"Individual Fold Validation Accuracies: {[f'{acc:.2f}%' for acc in fold_accuracy]}")
    print(f"Average Training Accuracy: {np.mean(fold_train_accuracy):.2f}% ± {np.std(fold_train_accuracy):.2f}%")
    print(f"Individual Fold Training Accuracies: {[f'{acc:.2f}%' for acc in fold_train_accuracy]}")


if __name__ == "__main__":
    main()