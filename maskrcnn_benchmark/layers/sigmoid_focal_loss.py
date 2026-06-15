# Modified to use pure PyTorch implementation instead of custom _C extension
import torch
from torch import nn
import torch.nn.functional as F


def sigmoid_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float,
    alpha: float,
    reduction: str = "none"
) -> torch.Tensor:
    """
    Sigmoid Focal Loss for dense object detection.
    
    Loss(x, class) = -alpha * (1-sigmoid(x))^gamma * log(sigmoid(x)) if class == 1
                   = -(1-alpha) * sigmoid(x)^gamma * log(1-sigmoid(x)) otherwise
    
    Arguments:
        logits (Tensor): (N, num_classes) float tensor of predicted logits
        targets (Tensor): (N,) tensor of target class indices (0 is background)
        gamma (float): Focusing parameter gamma >= 0
        alpha (float): Weighting factor in range (0,1) to balance positive vs negative
        reduction (str): 'none' | 'mean' | 'sum'
    
    Returns:
        Loss tensor
    """
    num_classes = logits.shape[1]
    
    # Handle gamma/alpha if they are tensors
    if isinstance(gamma, torch.Tensor):
        gamma = gamma.item() if gamma.numel() == 1 else gamma[0].item()
    if isinstance(alpha, torch.Tensor):
        alpha = alpha.item() if alpha.numel() == 1 else alpha[0].item()
    
    dtype = targets.dtype
    device = targets.device
    
    # Create class range [1, num_classes] for comparison
    # Background is class 0, foreground classes are 1 to num_classes
    class_range = torch.arange(1, num_classes + 1, dtype=dtype, device=device).unsqueeze(0)
    
    # Expand targets for broadcasting
    t = targets.unsqueeze(1)  # (N, 1)
    
    # Compute probabilities
    p = torch.sigmoid(logits)  # (N, num_classes)
    
    # Clamp for numerical stability
    p = torch.clamp(p, min=1e-7, max=1 - 1e-7)
    
    # Compute focal weights
    # For positive samples (t == class): weight = (1-p)^gamma
    # For negative samples (t != class): weight = p^gamma
    
    # term1: -(1-p)^gamma * log(p) for positive samples
    term1 = (1 - p).pow(gamma) * torch.log(p)
    
    # term2: -p^gamma * log(1-p) for negative samples
    term2 = p.pow(gamma) * torch.log(1 - p)
    
    # Positive mask: target equals this class
    pos_mask = (t == class_range).float()
    
    # Negative mask: target doesn't equal this class AND target >= 0 (valid)
    neg_mask = ((t != class_range) & (t >= 0)).float()
    
    # Compute loss
    loss = -pos_mask * term1 * alpha - neg_mask * term2 * (1 - alpha)
    
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss


# Keep backwards compatibility names
sigmoid_focal_loss_cuda = sigmoid_focal_loss
sigmoid_focal_loss_cpu = sigmoid_focal_loss


class SigmoidFocalLoss(nn.Module):
    """
    Sigmoid Focal Loss module.
    
    This is the loss function from "Focal Loss for Dense Object Detection"
    https://arxiv.org/abs/1708.02002
    
    Arguments:
        gamma (float): Focusing parameter. gamma=0 is equivalent to BCE loss.
        alpha (float): Balancing factor for positive/negative samples.
    """
    
    def __init__(self, gamma, alpha):
        super(SigmoidFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        """
        Arguments:
            logits (Tensor): (N, num_classes) predictions
            targets (Tensor): (N,) ground truth class indices
        
        Returns:
            Scalar loss tensor
        """
        loss = sigmoid_focal_loss(logits, targets, self.gamma, self.alpha)
        return loss.sum()

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "gamma=" + str(self.gamma)
        tmpstr += ", alpha=" + str(self.alpha)
        tmpstr += ")"
        return tmpstr
