
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PerClassFocalLoss(nn.Module):
    """
    Focal Loss with per-class alpha values for handling class-specific imbalance.

    This implementation computes a separate alpha weight for each class based on
    its frequency in the training set, addressing the issue where different classes
    have vastly different sample counts (e.g., 600 vs 33,300).

    Mathematical formulation:
        FL(p_t) = -α_c * (1 - p_t)^γ * log(p_t)

    where α_c is class-specific (different for each of the K classes).

    Args:
        alpha: Tensor of shape (num_classes,) with per-class weights, or
               float for uniform weighting (not recommended for imbalanced data)
        gamma: Focusing parameter (default: 2.0)
        reduction: 'mean' or 'sum'
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(PerClassFocalLoss, self).__init__()

        if isinstance(alpha, (float, int)):
            # If single value, use same alpha for all classes (not recommended)
            self.alpha = alpha
        else:
            # Per-class alpha values
            self.alpha = alpha

        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: (batch_size, num_classes) - logits
            targets: (batch_size, num_classes) - binary labels (0 or 1)

        Returns:
            Focal loss value
        """
        # Get probabilities
        p = torch.sigmoid(inputs)

        # Calculate BCE loss component (without reduction)
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )

        # Calculate p_t (probability of true class)
        p_t = p * targets + (1 - p) * (1 - targets)

        # Calculate focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # Apply per-class alpha weighting
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                # Uniform alpha
                alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            else:
                # Per-class alpha (recommended for imbalanced multi-label)
                # self.alpha should be shape (num_classes,)
                alpha = self.alpha.to(inputs.device)

                # For positive samples, use alpha[c]
                # For negative samples, use (1 - alpha[c])
                alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        else:
            alpha_t = 1.0

        # Combine all components
        focal_loss = alpha_t * focal_weight * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# ============================================================================
# ALPHA COMPUTATION METHODS
# ============================================================================

def compute_per_class_alpha_from_frequencies(class_frequencies, method='inverse', beta=0.999):
    """
    Compute per-class alpha values based on class frequencies.

    Args:
        class_frequencies: Array of shape (num_classes,) with counts for each class
        method: How to compute alpha
            - 'inverse': α_c = 1 / f_c (inverse frequency)
            - ' ': α_c = (1/f_c) / sum(1/f_c) (normalized)
            - 'effective_number': α_c = (1-β^f_c) / (1-β) (effective number of samples)
            - 'sqrt_inverse': α_c = 1 / sqrt(f_c) (square root of inverse)
            - 'balanced': α_c = N / (K * f_c) (balanced weighting)
        beta: For effective_number method (default: 0.999)

    Returns:
        alpha: Tensor of shape (num_classes,)
    """
    class_frequencies = np.array(class_frequencies, dtype=np.float32)
    num_classes = len(class_frequencies)
    total_samples = class_frequencies.sum()

    if method == 'inverse':
        # Simple inverse frequency
        alpha = 1.0 / (class_frequencies + 1e-6)

    elif method == 'inverse_normalized':
        # Normalized inverse frequency
        alpha = 1.0 / (class_frequencies + 1e-6)
        alpha = alpha / alpha.sum()

    elif method == 'effective_number':
        # Effective Number of Samples (Cui et al., 2019)
        # α_c = (1 - β^n_c) / (1 - β)
        # where β is typically 0.999 or 0.9999
        effective_num = 1.0 - np.power(beta, class_frequencies)
        alpha = (1.0 - beta) / (effective_num + 1e-6)

    elif method == 'sqrt_inverse':
        # Square root of inverse frequency (less aggressive)
        alpha = 1.0 / np.sqrt(class_frequencies + 1e-6)

    elif method == 'balanced':
        # Sklearn-style balanced weighting
        # α_c = n_samples / (n_classes * n_samples_c)
        alpha = total_samples / (num_classes * class_frequencies + 1e-6)

    else:
        raise ValueError(f"Unknown method: {method}")

    # Normalize to reasonable range (optional)
    # This prevents extreme values
    alpha = alpha / alpha.max()

    return torch.FloatTensor(alpha)