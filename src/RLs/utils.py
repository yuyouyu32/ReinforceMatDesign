import torch

def activate_A_func(a: torch.Tensor, k: int):
    
    values, indices = torch.topk(a.abs(), k)
    
    mask = torch.zeros_like(a).scatter_(0, indices, 1.0)
    
    activated = a * mask
    
    activated -= activated.mean()
    
    return activated
