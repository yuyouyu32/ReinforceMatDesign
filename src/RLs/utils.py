import torch
import numpy as np

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

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


class OrnsteinUhlenbeckNoise:
    def __init__(self, mu):
        self.theta, self.dt, self.sigma = 0.1, 0.01, 0.1
        self.mu = mu
        self.x_prev = np.zeros_like(self.mu)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x
    
    def reset(self):
        self.x_prev = np.zeros_like(self.mu)

def unit_test():
    batch_size = 3
    feature_size = 7
    a = torch.randn(batch_size, feature_size)
    k = torch.tensor([3, 4, 5])

    activated = activate_A_func(a, k)
    print(activated)
    print(sum(activated[0]))
    ou_noise = OrnsteinUhlenbeckNoise(mu=np.zeros(1))
    for i in range(10):
        print(ou_noise())
    
# python -m RLs.utils
if __name__ == '__main__':
    unit_test()