#!/usr/bin/env python3
"""
Main script for training and analyzing the Gaze Transformer model for 
Parkinson's Disease classification from eye movement data.

This script demonstrates the complete pipeline:
1. Data loading and preprocessing
2. Model training with cross-validation
3. Attention visualization and analysis
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

from model_implementation import GazeTransformer
from data_processing import load_data
from training import cross_validation_training
from utils import (
    visualize_attention_weights, 
    compare_attention_patterns,
    extract_all_attention_weights,
    plot_attention_comparison,
    plot_mean_attention_patterns
)


def run_training():
    """
    Run the complete training pipeline.
    """
    print("Starting Gaze Transformer Training Pipeline")
    print("=" * 60)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    print("\nLoading data...")
    eye, label = load_data('train_transformer.pkl', 'test_transformer.pkl')
    print(f"Data loaded successfully. Shape: {eye.shape}")
    
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
    
    print("\nModel Configuration:")
    for key, value in model_params.items():
        print(f"  {key}: {value}")
    
    print("\nTraining Configuration:")
    for key, value in training_params.items():
        print(f"  {key}: {value}")
    
    # Perform cross-validation training
    print("\nStarting cross-validation training...")
    fold_accuracy, fold_train_accuracy = cross_validation_training(
        eye, label, model_params, training_params
    )
    
    # Print final results
    print(f"\n{'='*60}")
    print("TRAINING COMPLETED")
    print(f"{'='*60}")
    print(f"Average Validation Accuracy: {np.mean(fold_accuracy):.2f}% ± {np.std(fold_accuracy):.2f}%")
    print(f"Individual Fold Validation Accuracies: {[f'{acc:.2f}%' for acc in fold_accuracy]}")
    print(f"Average Training Accuracy: {np.mean(fold_train_accuracy):.2f}% ± {np.std(fold_train_accuracy):.2f}%")
    print(f"Individual Fold Training Accuracies: {[f'{acc:.2f}%' for acc in fold_train_accuracy]}")
    
    return eye, label, model_params, device


def run_analysis(eye, label, model_params, device):
    """
    Run attention analysis and visualization.
    
    Args:
        eye: Eye movement data
        label: Labels
        model_params: Model parameters
        device: Computing device
    """
    print("\n" + "="*60)
    print("ATTENTION ANALYSIS")
    print("="*60)
    
    # Load the trained model
    model = GazeTransformer(
        model_params['circle_length'],
        model_params['num_patches'],
        model_params['embedding_dim'],
        model_params['num_heads'],
        model_params['num_layers'],
        model_params['num_classes']
    ).to(device)
    
    try:
        model.load_state_dict(torch.load("PD_Gaze_Cls.pth", map_location=device))
        print("Model loaded successfully!")
    except FileNotFoundError:
        print("Warning: Could not find saved model. Using randomly initialized weights.")
    
    # Visualize attention for specific subjects
    print("\n1. Visualizing attention weights for individual subjects...")
    
    # Example subjects (you may want to choose specific indices)
    hc_subject_idx = 84  # HC subject
    pd_subject_idx = 36  # PD subject
    
    print(f"Visualizing HC subject (index {hc_subject_idx})...")
    visualize_attention_weights(
        eye, label, model, hc_subject_idx, 
        model_params['circle_length'], device
    )
    
    print(f"Visualizing PD subject (index {pd_subject_idx})...")
    visualize_attention_weights(
        eye, label, model, pd_subject_idx, 
        model_params['circle_length'], device
    )
    
    # Compare attention patterns between HC and PD
    print("\n2. Comparing attention patterns between HC and PD subjects...")
    compare_attention_patterns(
        eye, label, model, [hc_subject_idx, pd_subject_idx], 
        model_params['circle_length'], device
    )
    
    # Extract attention weights for all subjects
    print("\n3. Extracting attention weights for all subjects...")
    hc_weights, pd_weights = extract_all_attention_weights(eye, label, model, device)
    
    # Plot attention comparison
    print("\n4. Plotting attention weight comparison...")
    plot_attention_comparison(hc_weights, pd_weights)
    
    # Plot mean attention patterns
    print("\n5. Plotting mean attention patterns...")
    # Filter weights first
    hc_weights_filtered = [np.array(hw) for hw in hc_weights if np.max(hw) < 0.2]
    pd_weights_filtered = [np.array(pw) for pw in pd_weights if np.max(pw) < 0.5]
    
    plot_mean_attention_patterns(hc_weights_filtered, pd_weights_filtered)
    
    print("\nAttention analysis completed!")


def main():
    """
    Main function that runs the complete pipeline.
    """
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("Gaze Transformer for Parkinson's Disease Classification")
    print("Using Deformable Token Embedding and Attention Analysis")
    print("="*60)
    
    # Run training
    eye, label, model_params, device = run_training()
    
    # Run analysis
    run_analysis(eye, label, model_params, device)
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nGenerated files:")
    print("- PD_Gaze_Cls.pth: Final trained model")
    print("- best_model_fold_*.pth: Best models for each fold")
    print("- attention_weights.svg: Attention visualization")


if __name__ == "__main__":
    main()
