import numpy as np

from config import *
from env.env_config import *
from env.ML_model import ML_Model


class Enviroment:
    def __init__(self) -> None:
        self.models = ML_Model(DataPath, DropColumns, TargetColumns, MLResultPath)
        self.best_models = self.models.get_best_models()
    
    def target_func(self, s):
        X = np.array(s).reshape(1, -1)
        results = {}
        for target, model in self.best_models.items():
            results[target] = model.predict(X)[0]
        return results
    
    def step(self, s, a, indexs):
        s_ = s.copy()
        assert len(a) == len(indexs)
        s_[indexs] = s_[indexs] + a
        return s_
    
    def reset_random_state(self):
        s = StartPool[np.random.randint(0, StartPool.shape[0])]
        indexs = np.where(s != 0)[0]
        noise = np.random.rand(len(indexs)) * A_Scale
        noise = noise - noise.mean()
        return self.step(s, noise, indexs)

    def get_BMG_str(self, s):
        BMG_str = ''
        indexs = np.where(s != 0)[0]
        indexs = indexs[np.argsort(-s[indexs])]
        for index in indexs:
            BMG_str += CompositionClomuns[index]
            BMG_str += str(int(s[index]))
        return BMG_str
        
    def judege_s(self, s):
        if abs(sum(s) - 100) > 0.1 or s.min() < 0 or s.max() > 100:
            return False
        return True
    
    def judge_a(self, a):
        if a.min() < -A_Scale or a.max() > A_Scale:
            return False
        return True

    def reward(self, s, a, s_):
        result = self.target_func(s)
        result_ = self.target_func(s_)
        print(result,result_)
        # CL Phase 1 (teach model to generate the legal action)
        if not self.judege_s(s) or not self.judege_s(s_) or not self.judge_a(a):
            r = -1
        else:
            r = 0
            
        return r

def unit_test():
    env = Enviroment()
    s = env.reset_random_state()
    print(env.get_BMG_str(s))
    indexs = np.where(s != 0)[0]
    a = np.random.rand(len(indexs)) * A_Scale
    a = a - a.mean()
    s_ = env.step(s, a, indexs)
    r = env.reward(s, a, s_)
    print(env.get_BMG_str(s_))
    print(r)
    
if __name__ == '__main__':
    unit_test()
