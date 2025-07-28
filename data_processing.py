import torch
import numpy as np
import pickle
from torch.utils.data import DataLoader, TensorDataset


def load_data(train_path='train_transformer.pkl', test_path='test_transformer.pkl'):
    """
    Load training and testing data from pickle files.
    
    Args:
        train_path: Path to training data pickle file
        test_path: Path to testing data pickle file
    
    Returns:
        eye: Combined eye movement data tensor
        label: Combined label tensor
    """
    # Load data
    train_data = pickle.load(open(train_path, 'rb'))
    test_data = pickle.load(open(test_path, 'rb'))
    
    print(f"Training eye data shape: {np.array(train_data['eye']).shape}")
    
    # Convert to tensors
    train_eye = torch.tensor(np.array(train_data['eye']), dtype=torch.float32)
    train_label = torch.tensor(np.array(train_data['label']), dtype=torch.long)
    
    test_eye = torch.tensor(np.array(test_data['eye']), dtype=torch.float32)
    test_label = torch.tensor(np.array(test_data['label']), dtype=torch.long)
    
    # Combine training and testing data
    eye = torch.cat((train_eye, test_eye), 0)
    label = torch.cat((train_label, test_label), 0)
    
    return eye, label


def bootstrap_balance_minority(data, labels):
    """
    Balance dataset by bootstrapping the minority class.
    
    Args:
        data: Input data tensor
        labels: Label tensor
    
    Returns:
        balanced_data: Balanced data tensor
        balanced_labels: Balanced label tensor
    """
    class_0_indices = (labels == 0).nonzero(as_tuple=True)[0]
    class_1_indices = (labels == 1).nonzero(as_tuple=True)[0]

    if len(class_0_indices) < len(class_1_indices):
        minority_indices = class_0_indices
        majority_indices = class_1_indices
    else:
        minority_indices = class_1_indices
        majority_indices = class_0_indices
    
    num_minority_samples_needed = len(majority_indices) - len(minority_indices)
    additional_samples = np.random.choice(
        minority_indices, size=num_minority_samples_needed, replace=True
    )
    
    balanced_minority_indices = torch.cat((minority_indices, torch.tensor(additional_samples)))
    balanced_indices = torch.cat((balanced_minority_indices, majority_indices))
    balanced_indices = balanced_indices[torch.randperm(len(balanced_indices))]

    balanced_data = data[balanced_indices]
    balanced_labels = labels[balanced_indices]

    return balanced_data, balanced_labels


def create_data_loaders(train_data, train_labels, val_data, val_labels, 
                       train_batch_size=8, val_batch_size=32):
    """
    Create PyTorch data loaders for training and validation.
    
    Args:
        train_data: Training data tensor
        train_labels: Training labels tensor
        val_data: Validation data tensor
        val_labels: Validation labels tensor
        train_batch_size: Batch size for training
        val_batch_size: Batch size for validation
    
    Returns:
        train_loader: Training data loader
        val_loader: Validation data loader
    """
    train_dataset = TensorDataset(train_data, train_labels)
    val_dataset = TensorDataset(val_data, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)
    
    return train_loader, val_loader


def prepare_fold_data(eye, label, train_index, test_index, balance_data=True):
    """
    Prepare data for a specific fold in cross-validation.
    
    Args:
        eye: Complete eye movement data
        label: Complete labels
        train_index: Indices for training data
        test_index: Indices for test data
        balance_data: Whether to balance the minority class
    
    Returns:
        train_eye: Training eye data
        train_label: Training labels
        test_eye: Test eye data
        test_label: Test labels
    """
    train_eye, test_eye = eye[train_index], eye[test_index]
    train_label, test_label = label[train_index], label[test_index]
    
    if balance_data:
        # Bootstrap balance minority class
        train_eye, train_label = bootstrap_balance_minority(train_eye, train_label)
        test_eye, test_label = bootstrap_balance_minority(test_eye, test_label)
    
    return train_eye, train_label, test_eye, test_label
