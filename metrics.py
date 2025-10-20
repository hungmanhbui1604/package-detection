import torch

def MCE(y_pred, y_true, threshold=0.05):
    e = torch.sqrt(((y_true - y_pred) ** 2).sum(dim=1))  # (N,)
    normalized_e = e / threshold
    mce = torch.minimum(normalized_e, torch.ones_like(normalized_e))  # element-wise min
    return mce.mean()

def OE(y_pred, y_true, theta_max=torch.pi/9):
    # Normalize vectors
    norm_pred = torch.norm(y_pred, dim=1) # (N,)
    norm_true = torch.norm(y_true, dim=1)
    dot = torch.sum(y_true * y_pred, dim=1) # (N,)
    cos_theta = dot / (norm_pred * norm_true + 1e-8)
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    theta = torch.acos(cos_theta)
    
    normalized_theta = theta / theta_max
    oe = torch.minimum(normalized_theta, torch.ones_like(normalized_theta))
    return oe.mean()

def AC(mce, oe, lambda1=0.7, lambda2=0.3):
    ac = (1.0 - mce) * lambda1 + (1.0 - oe) * lambda2
    return ac