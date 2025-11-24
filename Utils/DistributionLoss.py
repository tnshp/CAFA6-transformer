import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResampleLoss(nn.Module):
    """
    Distribution-Balanced Loss (DB Loss) - CORRECTED VERSION
    
    Fixes:
    - Numerical stability using logsigmoid instead of log(1 + exp(x))
    - Proper loss scaling (mean instead of sum/num_classes)
    - Weight clamping to prevent extreme values
    - Gradient stability improvements
    """
    
    def __init__(
        self,
        freq_file=None,
        class_freq=None,
        num_classes=80,
        use_rebalance=True,
        use_negative_tolerant=True,
        reweight_func='rebalance',
        focal_gamma=0.0,
        focal_balance_param=2.0,
        cb_beta=0.9999,
        map_alpha=0.1,
        map_beta=10.0,
        map_mu=0.2,
        lambda_neg=2.0,
        kappa=0.05,
        reduction='mean'
    ):
        super(ResampleLoss, self).__init__()
        
        assert reweight_func in ['rebalance', 'inverse', 'sqrt_inv', None]
        
        self.num_classes = num_classes
        self.use_rebalance = use_rebalance
        self.use_negative_tolerant = use_negative_tolerant
        self.reweight_func = reweight_func
        self.focal_gamma = focal_gamma
        self.focal_balance_param = focal_balance_param
        self.cb_beta = cb_beta
        self.reduction = reduction
        
        # Re-balanced weighting parameters
        self.map_alpha = map_alpha
        self.map_beta = map_beta
        self.map_mu = map_mu
        
        # Negative-tolerant regularization parameters
        self.lambda_neg = lambda_neg
        self.kappa = kappa
        
        # Load or set class frequencies
        if freq_file is not None:
            if freq_file.endswith('.pkl'):
                import pickle
                with open(freq_file, 'rb') as f:
                    class_freq = pickle.load(f)
            elif freq_file.endswith('.npy'):
                class_freq = np.load(freq_file)
            else:
                raise ValueError("freq_file must be .pkl or .npy")
        
        if class_freq is not None:
            if isinstance(class_freq, list):
                class_freq = np.array(class_freq)
            self.class_freq = torch.from_numpy(class_freq).float()
        else:
            # If no frequency provided, use uniform
            self.class_freq = torch.ones(num_classes).float()
        
        # Initialize class-specific bias for NTR
        if self.use_negative_tolerant:
            self.init_bias = self.get_init_bias(self.class_freq)
        else:
            self.init_bias = None
    
    def get_init_bias(self, class_freq):
        """Calculate class-specific bias initialization for NTR"""
        # Calculate class prior: p_i = n_i / N
        total_samples = class_freq.sum()
        class_prior = class_freq / total_samples
        
        # Clamp to avoid extreme values
        class_prior = torch.clamp(class_prior, min=1e-4, max=1-1e-4)
        
        # Calculate optimal bias: b_i = -log(1/p_i - 1)
        init_bias = -torch.log((1.0 / class_prior) - 1.0)
        
        # Apply scale factor κ: ν_i = -κ * b_i
        init_bias = -self.kappa * init_bias
        
        # Clamp to reasonable range
        init_bias = torch.clamp(init_bias, min=-10.0, max=10.0)
        
        return init_bias
    
    def get_rebalance_weight(self, gt_labels):
        """Calculate re-balanced weight for each instance"""
        # Get per-class inverse frequency: 1/n_i
        class_freq = self.class_freq.to(gt_labels.device)
        per_class_weights = 1.0 / (class_freq + 1.0)  # Add 1 to avoid extreme weights
        
        # Calculate instance-level sampling probability
        instance_weights = torch.sum(
            gt_labels * per_class_weights.unsqueeze(0),
            dim=1,
            keepdim=True
        )  # Shape: (B, 1)
        
        # Avoid division by zero
        instance_weights = torch.clamp(instance_weights, min=1e-6)
        
        # Calculate re-balancing weight
        rebalance_w = per_class_weights.unsqueeze(0) / instance_weights
        
        # Clamp before sigmoid to avoid extreme values
        rebalance_w = torch.clamp(rebalance_w, min=0.0, max=10.0)
        
        # Apply smoothing function
        rebalance_w = self.map_alpha + torch.sigmoid(
            self.map_beta * (rebalance_w - self.map_mu)
        )
        
        # Final clamping of weights
        rebalance_w = torch.clamp(rebalance_w, min=0.1, max=10.0)
        
        return rebalance_w
    
    def negative_tolerant_loss(self, logits, labels, weights):
        """
        Calculate negative-tolerant regularized loss with numerical stability
        """
        # Get bias initialization
        init_bias = self.init_bias.to(logits.device).unsqueeze(0)
        
        # Shift logits by bias: z - ν
        shifted_logits = logits - init_bias
        
        # Clamp logits to prevent overflow
        shifted_logits = torch.clamp(shifted_logits, min=-50.0, max=50.0)
        
        # Use F.softplus for numerical stability: softplus(x) = log(1 + exp(x))
        # Positive loss: y * softplus(-z_shifted)
        pos_loss = labels * F.softplus(-shifted_logits)
        
        # Negative loss with scaling: (1/λ) * (1-y) * softplus(-λ*z_shifted)
        neg_loss = (1.0 / self.lambda_neg) * (1 - labels) * F.softplus(
            -self.lambda_neg * shifted_logits
        )
        
        # Combine with weights
        loss = weights * (pos_loss + neg_loss)
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
    
    def standard_bce_loss(self, logits, labels, weights):
        """
        Calculate standard weighted BCE loss with numerical stability
        """
        # Clamp logits
        logits = torch.clamp(logits, min=-50.0, max=50.0)
        
        # Use F.softplus for numerical stability
        # BCE: y*log(1+exp(-z)) + (1-y)*log(1+exp(z))
        #    = y*softplus(-z) + (1-y)*softplus(z)
        pos_loss = labels * F.softplus(-logits)
        neg_loss = (1 - labels) * F.softplus(logits)
        
        loss = weights * (pos_loss + neg_loss)
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
    
    def focal_loss_modulation(self, logits, labels):
        """Apply focal loss modulation factor"""
        # Calculate probabilities with clamping
        logits = torch.clamp(logits, min=-50.0, max=50.0)
        probs = torch.sigmoid(logits)
        probs = torch.clamp(probs, min=1e-7, max=1-1e-7)
        
        # Calculate pt: p if y=1, else 1-p
        pt = probs * labels + (1 - probs) * (1 - labels)
        pt = torch.clamp(pt, min=1e-7, max=1-1e-7)
        
        # Focal weight: (1 - pt)^γ
        focal_weight = torch.pow(1 - pt, self.focal_gamma)
        
        # Apply balance parameter
        alpha_t = self.focal_balance_param * labels + (1 - labels)
        focal_weight = alpha_t * focal_weight
        
        return focal_weight
    
    def forward(self, logits, labels):
        """
        Forward pass of Distribution-Balanced Loss
        
        Parameters
        ----------
        logits : torch.Tensor
            Model output logits of shape (batch_size, num_classes)
        labels : torch.Tensor
            Ground truth binary targets of shape (batch_size, num_classes)
        
        Returns
        -------
        loss : torch.Tensor
            Scalar loss value
        """
        # Ensure labels are float
        labels = labels.float()
        
        # Calculate weights
        if self.use_rebalance and self.reweight_func == 'rebalance':
            weights = self.get_rebalance_weight(labels)
        else:
            weights = torch.ones_like(labels)
        
        # Calculate base loss
        if self.use_negative_tolerant:
            loss = self.negative_tolerant_loss(logits, labels, weights)
        else:
            loss = self.standard_bce_loss(logits, labels, weights)
        
        # Apply focal loss modulation if enabled
        if self.focal_gamma > 0:
            # Note: For simplicity, we don't recalculate with focal weights
            # as it requires element-wise loss computation
            # You can implement this if needed
            pass
        
        return loss


class DistributionBalancedLoss(nn.Module):
    """
    Simplified Distribution-Balanced Loss interface - CORRECTED VERSION
    """
    
    def __init__(
        self,
        class_freq,
        num_classes=64,
        alpha=0.1,
        beta=10.0,
        mu=0.2,
        lambda_neg=2.0,
        kappa=0.05,
        reduction='mean'
    ):
        """
        Parameters
        ----------
        class_freq : list or array
            Frequencies of each class [n_1, n_2, ..., n_C]
        num_classes : int
            Number of classes
        alpha : float
            Overall lift for smoothing function (0.1 to 0.5 recommended)
        beta : float
            Shape parameter for smoothing function (10.0 default)
        mu : float
            Center parameter for smoothing function (0.2 for many classes, 0.3 for fewer)
        lambda_neg : float
            Negative sample gradient scale (2.0 to 5.0, higher for more imbalance)
        kappa : float
            Bias initialization scale (0.05 to 0.1)
        reduction : str
            'mean' or 'sum'
        """
        super(DistributionBalancedLoss, self).__init__()
        
        self.loss = ResampleLoss(
            class_freq=class_freq,
            num_classes=num_classes,
            use_rebalance=True,
            use_negative_tolerant=True,
            reweight_func='rebalance',
            map_alpha=alpha,
            map_beta=beta,
            map_mu=mu,
            lambda_neg=lambda_neg,
            kappa=kappa,
            reduction=reduction
        )
    
    def forward(self, logits, labels):
        return self.loss(logits, labels)


class DistributionBalancedLossSimple(nn.Module):
    """
    Simplified version for debugging - uses only basic reweighting
    """
    
    def __init__(self,
        class_freq,
        num_classes=64,
        alpha=0.1,
        beta=10.0,
        mu=0.2,
        lambda_neg=2.0,
        kappa=0.05,
        reduction='mean'
        ):
        super(DistributionBalancedLossSimple, self).__init__()
        
        if isinstance(class_freq, list):
            class_freq = np.array(class_freq)
        
        self.class_freq = torch.from_numpy(class_freq).float()
        self.num_classes = num_classes
        
        # Calculate simple inverse frequency weights
        total_samples = self.class_freq.sum()
        self.weights = total_samples / (self.num_classes * self.class_freq)
        
        # Normalize weights to have mean of 1.0
        self.weights = self.weights / self.weights.mean()
        
        # Clamp to reasonable range
        self.weights = torch.clamp(self.weights, min=0.1, max=10.0)
    
    def forward(self, logits, labels):
        """
        Simple weighted BCE loss with numerical stability
        """
        labels = labels.float()
        weights = self.weights.to(logits.device)
        
        # Clamp logits
        logits = torch.clamp(logits, min=-50.0, max=50.0)
        
        # Use BCEWithLogitsLoss for numerical stability
        # This internally uses log_sigmoid for better numerical properties
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, 
            labels, 
            weight=weights.unsqueeze(0),
            reduction='mean'
        )
        
        return bce_loss
