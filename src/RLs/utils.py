import torch
import torch.nn as nn

def activate_A_func(a: torch.Tensor, k: torch.Tensor):
    batch_size, feature_size = a.size()
    activated = torch.zeros_like(a)
    for i in range(batch_size):
        top_k_indices = torch.arange(k[i].item())
        
        mask = torch.zeros_like(a[i])
        mask[top_k_indices] = 1.0
        
        top_k_values = a[i] * mask
        
        top_k_sum = top_k_values.sum()
        if k[i].item() > 0:
            adjustment = top_k_sum / k[i].item()
            top_k_values[top_k_indices] -= adjustment
            
        activated[i] = top_k_values
    return activated

def unit_test():
    batch_size = 3
    feature_size = 7
    a = torch.randn(batch_size, feature_size)
    k = torch.tensor([3, 4, 5])

    activated = activate_A_func(a, k)
    print(activated)
    print(sum(activated[0]))
    
# python -m RLs.utils
if __name__ == '__main__':
    unit_test()