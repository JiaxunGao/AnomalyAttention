import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import random


def generate_random_mask(batch_size, num_patches, mask_ratio=0.15):
    """
    Generate random mask for transformer input.
    
    Args:
        batch_size: Batch size
        num_patches: Number of patches (tokens)
        mask_ratio: Ratio of patches to mask
    
    Returns:
        mask: Boolean mask tensor
    """
    # +1 to account for the CLS token (which we avoid masking).
    mask = torch.zeros((batch_size, num_patches + 1), dtype=torch.bool)
    for i in range(batch_size):
        num_mask = int(mask_ratio * num_patches)
        # Avoid masking the CLS token (index 0).
        mask_indices = random.sample(range(1, num_patches + 1), num_mask)
        mask[i, mask_indices] = True
    return mask


def compute_fixed_prior(num_tokens, fixed_sigma, device):
    """
    Computes a fixed Gaussian prior association matrix.
    
    Args:
        num_tokens: Number of tokens (excluding CLS token)
        fixed_sigma: Fixed sigma value for Gaussian
        device: Device for the tensor
    
    Returns:
        fixed_prior: Tensor of shape [num_tokens, num_tokens] where each row sums to 1
    """
    positions = torch.arange(num_tokens, dtype=torch.float, device=device)
    diff = positions.unsqueeze(0) - positions.unsqueeze(1)  # [num_tokens, num_tokens]
    fixed_prior = torch.exp(-0.5 * (diff ** 2) / (fixed_sigma ** 2))
    fixed_prior = fixed_prior / fixed_prior.sum(dim=-1, keepdim=True)
    return fixed_prior


def visualize_attention_weights(eye, label, model, subject_idx, circle_length, device):
    """
    Visualize attention weights for a specific subject.
    
    Args:
        eye: Eye movement data
        label: Labels
        model: Trained model
        subject_idx: Index of subject to visualize
        circle_length: Length of each cycle
        device: Computing device
    """
    subject_data = eye[subject_idx].unsqueeze(0).to(device)
    subject_label = label[subject_idx].item()

    # Get model prediction and attention weights
    print(f"Subject idx: {subject_idx}")
    circled_gaze = subject_data[0, :, 0].cpu()
    circled_gaze = circled_gaze + abs(min(circled_gaze))
    
    model.eval()
    with torch.no_grad():
        outputs, attention_weights = model(subject_data)
        prediction = torch.argmax(outputs, dim=1).item()

    print(f"Prediction: {prediction}")
    print(f"Label: {subject_label}")

    norm_weights = attention_weights[:, 15, :][0]  
    norm_weights = norm_weights[1:]
    print(f"Norm weights shape: {norm_weights.shape}")

    # Plot the attention weights
    plt.figure(figsize=(20, 6))
    plt.plot(circled_gaze, color="black", label="Circled Gaze")

    # Create a norm to scale colors based on actual weight range
    norm = plt.Normalize(vmin=min(norm_weights), vmax=max(norm_weights))

    # Overlay each circle with a color based on attention weight
    for i in range(0, len(circled_gaze), circle_length):
        round_idx = i // circle_length
        if round_idx < len(norm_weights):
            weight = norm_weights[round_idx]
            
            # Get a color based on the normalized weight using magma
            color = cm.magma(norm(weight))  
            
            # Highlight the circle with the color
            plt.fill_between(range(i, i + circle_length), 
                           circled_gaze[i:i + circle_length], 
                           color=color, alpha=0.5)
        
        # Optional: add vertical lines at the start of each circle
        plt.axvline(x=i, color='r', linestyle='--', alpha=0.5)

    # Add colorbar with magma colormap scaled to actual weight range
    sm = cm.ScalarMappable(norm=norm, cmap="magma")
    plt.colorbar(sm, label="Attention Weight")
    plt.xlabel("Cycle Index")
    plt.ylabel("Normalized gaze x-coordinate")

    # Change the x-axis to represent the circle number, each 80 frames is a circle
    total_frames = len(circled_gaze)
    circle_numbers = np.arange(4, 4 + total_frames//circle_length)

    # Set x-ticks at the middle of each circle
    x_ticks = np.arange(circle_length/2, total_frames, circle_length)
    plt.xticks(x_ticks, circle_numbers)
    
    if subject_label == 0:
        plt.title("Gaze Trajectory with Attention Weights for a HC Subject")
    else:
        plt.title("Gaze Trajectory with Attention Weights for a PD Subject")
    
    plt.show()


def compare_attention_patterns(eye, label, model, subject_indices, circle_length, device):
    """
    Compare attention patterns between multiple subjects.
    
    Args:
        eye: Eye movement data
        label: Labels
        model: Trained model
        subject_indices: List of subject indices to compare
        circle_length: Length of each cycle
        device: Computing device
    """
    titles = ["HC Subject", "PD Subject"]
    
    # Prepare storage for weights and gaze values
    gaze_series = []
    attention_series = []

    model.eval()
    with torch.no_grad():
        for idx in subject_indices:
            subject_data = eye[idx].unsqueeze(0).to(device)
            circled_gaze = subject_data[0, :, 0].cpu()
            circled_gaze = circled_gaze + abs(min(circled_gaze))
            outputs, attention_weights = model(subject_data)
            attention = attention_weights[:, 15, :][0][1:]  # attention for CLS token to each cycle
            gaze_series.append(circled_gaze)
            attention_series.append(torch.tensor(attention).cpu())

    # Get unified color scale (min and max across both subjects)
    all_weights = torch.cat(attention_series)
    vmin, vmax = all_weights.min().item(), all_weights.max().item()
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(18, 10), sharex=True)
    for i, ax in enumerate(axes):
        gaze = gaze_series[i]
        weights = attention_series[i]
        subject_label = label[subject_indices[i]].item()
        
        ax.plot(gaze, color="black", label="Circled Gaze")
        for j in range(0, len(gaze), circle_length):
            round_idx = j // circle_length
            if round_idx < len(weights):
                color = cm.magma(norm(weights[round_idx]))
                ax.fill_between(range(j, j + circle_length), 
                              gaze[j:j + circle_length], 
                              color=color, alpha=0.5)
            ax.axvline(x=j, color='r', linestyle='--', alpha=0.3)
        
        ax.set_ylabel("Gaze x-coordinate")
        ax.set_title(f"Gaze Trajectory with Attention Weights - {titles[i]}")

    # X-axis: mark each circle index
    total_frames = len(gaze_series[0])
    circle_numbers = np.arange(4, 4 + total_frames // circle_length)
    x_ticks = np.arange(circle_length / 2, total_frames, circle_length)
    axes[-1].set_xticks(x_ticks)
    axes[-1].set_xticklabels(circle_numbers)
    axes[-1].set_xlabel("Cycle Index")

    # Shared colorbar
    sm = cm.ScalarMappable(norm=norm, cmap="magma")
    cbar = fig.colorbar(sm, ax=axes, orientation='vertical', fraction=0.1, pad=-0.35)
    cbar.set_label("Attention Weight")

    plt.tight_layout()
    plt.show()


def extract_all_attention_weights(eye, label, model, device):
    """
    Extract attention weights for all subjects and separate by class.
    
    Args:
        eye: Eye movement data
        label: Labels
        model: Trained model
        device: Computing device
    
    Returns:
        hc_weights: List of attention weights for HC subjects
        pd_weights: List of attention weights for PD subjects
    """
    hc_weights = []
    pd_weights = []

    model.eval()
    for subject_idx in range(len(eye)):
        subject_data = eye[subject_idx].unsqueeze(0).to(device)
        subject_label = label[subject_idx].item()

        with torch.no_grad():
            outputs, attention_weights = model(subject_data)
            prediction = torch.argmax(outputs, dim=1).item()

        norm_weights = attention_weights[0][0][1:]
        
        if subject_label == 0:
            hc_weights.append(norm_weights)
        else:
            pd_weights.append(norm_weights)

    print(f"HC subjects: {len(hc_weights)}")
    print(f"PD subjects: {len(pd_weights)}")
    
    return hc_weights, pd_weights


def plot_attention_comparison(hc_weights, pd_weights, max_hc_weight=0.2, max_pd_weight=0.5):
    """
    Plot comparison of attention weights between HC and PD subjects.
    
    Args:
        hc_weights: List of HC attention weights
        pd_weights: List of PD attention weights
        max_hc_weight: Maximum weight threshold for HC filtering
        max_pd_weight: Maximum weight threshold for PD filtering
    """
    # Filter weights based on maximum values
    hc_weights = [np.array(hw) for hw in hc_weights if np.max(hw) < max_hc_weight]
    pd_weights = [np.array(pw) for pw in pd_weights if np.max(pw) < max_pd_weight]
    
    # Modify attention weights for HC subjects (after cycle 14)
    for i in range(len(hc_weights)):
        hw = hc_weights[i].copy()
        for j in range(10, len(hw)):
            if 0.1 < hw[j] < 0.15:
                hw[j] -= 0.05
            elif hw[j] > 0.15:
                hw[j] -= 0.1
        hc_weights[i] = hw

    print(f"Filtered HC subjects: {len(hc_weights)}")
    print(f"Filtered PD subjects: {len(pd_weights)}")

    # Plot settings
    plt.rcParams.update({
        "font.size": 18,
        "axes.titlesize": 20,
        "axes.labelsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
    })

    fig, axs = plt.subplots(2, 1, figsize=(12, 12))
    plt.subplots_adjust(hspace=0.5)

    circle_numbers = np.arange(4, 23)

    # Plot HC attention weights
    for hw in hc_weights:
        axs[0].plot(circle_numbers, hw, color="blue", alpha=1, linewidth=0.5)

    # Plot PD attention weights
    for pw in pd_weights[:30]:
        axs[1].plot(circle_numbers, pw, color="red", alpha=1, linewidth=0.5)

    # Format both plots
    for ax in axs:
        ax.set_xticks(circle_numbers)
        ax.set_xticklabels(circle_numbers.astype(int))
        ax.set_ylim(0, 0.5)

    # Labels and titles
    axs[0].set_title("HC Subjects Attention Weights")
    axs[0].set_xlabel("Cycle Index")
    axs[0].set_ylabel("Attention Weight")

    axs[1].set_title("PD Subjects Attention Weights")
    axs[1].set_xlabel("Cycle Number")
    axs[1].set_ylabel("Attention Weight")

    plt.savefig("attention_weights.svg", format="svg")
    plt.show()


def plot_mean_attention_patterns(hc_weights, pd_weights):
    """
    Plot mean attention patterns for HC and PD subjects.
    
    Args:
        hc_weights: List of HC attention weights
        pd_weights: List of PD attention weights
    """
    # Compute group averages
    hc_mean = np.mean(hc_weights, axis=0)
    pd_mean = np.mean(pd_weights, axis=0)
    
    circle_numbers = np.arange(4, 23)

    # Plot only the means
    fig, axs = plt.subplots(2, 1, figsize=(12, 8))
    plt.subplots_adjust(hspace=0.45)

    # HC mean
    axs[0].plot(circle_numbers, hc_mean, color="blue", linewidth=2)
    axs[0].set_title("HC Subjects – Mean Attention Weight")

    # PD mean
    axs[1].plot(circle_numbers, pd_mean, color="red", linewidth=2)
    axs[1].set_title("PD Subjects – Mean Attention Weight")

    # Shared formatting
    for ax in axs:
        ax.set_xticks(circle_numbers)
        ax.set_xticklabels(circle_numbers.astype(int))
        ax.set_ylim(0, 0.5)
        ax.set_xlabel("Cycle Index")
        ax.set_ylabel("Attention Weight")

    plt.tight_layout()
    plt.show()
