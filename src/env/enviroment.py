import numpy as np

from config import *
from env.env_config import *
from env.ML_model import ML_Model
from BMGs import BMGs
from typing import List, Tuple, Dict, Set


class Enviroment:
    def __init__(self) -> None:
        self.models = ML_Model(DataPath, DropColumns, TargetColumns, MLResultPath)
        self.best_models = self.models.get_best_models()
        self.matrix_thresholds, self.exist_bmgs = self.define_matrix_thresholds(DataPath)
    
    @staticmethod
    def define_matrix_thresholds(excel_file: str, percentile: float = 0.8) -> Tuple[Dict, Set]:
        """
        Define the matrix thresholds for each target property.
        
        Parameters:
            excel_file (str): path to the Excel file
            percentile (float): percentile for the threshold
        
        Returns:
            matrix_thresholds (dict): matrix thresholds for each target property
            exist_bmgs (list): list of existing BMGs
        """
        df = pd.read_excel(excel_file)
        # Get the unique existing BMGs
        exist_bmgs = set(df['Chemical composition'].unique().tolist())
        # Identify the base matrix for each row
        df = df.drop(columns=DropColumns)
        df['Base_Matrix'] = df.drop(columns=TargetColumns).iloc[:, 1:].idxmax(axis=1)
        # Use the Base_Matrix values to groupby, then use the percentile to find the percentile of each group, only for TargetColumns, to get a thresholds dict
        matrix_thresholds = {group: data[TargetColumns].quantile(percentile).to_dict() for group, data in df.groupby('Base_Matrix')}
        matrix_thresholds = pd.DataFrame(matrix_thresholds).T
        # Add a new row for new BMGs with a specified data type
        matrix_thresholds.loc['New'] = pd.Series(dtype=float)
        # Fill the NaN values with the median of the column
        matrix_thresholds_filled = matrix_thresholds.apply(lambda x: x.fillna(x.median()), axis=0)
        

        matrix_thresholds_filled = matrix_thresholds_filled.to_dict(orient='index')

        return matrix_thresholds_filled, exist_bmgs
    
    def target_func(self, s: np.ndarray) -> Dict:
        """
        Get the target properties of a BMGs.
        
        Parameters:
            s (np.ndarray): state vector
        
        Returns:
            dict: target properties
        """
        X = np.array(s).reshape(1, -1)
        results = {}
        for target, model in self.best_models.items():
            results[target] = model.predict(X)[0]
        return results
    
    def step(self, s: np.ndarray, a: np.ndarray) -> np.ndarray:
        """
        Take a step in the environment.
        
        Parameters:
            s (np.ndarray): state vector
            a (np.ndarray): action vector
            indexs (np.ndarray): indices of the non-zero elements in the state vector
        
        Returns:
            np.ndarray: new state vector
        """
        s_ = s.copy()
        indexs = np.where(s != 0)[0]
        assert len(a) == len(indexs)
        s_[indexs] = s_[indexs] + a
        return s_
    
    def reset_random_state(self):
        """
        Reset the state vector randomly.
        
        Returns:
            np.ndarray: state vector
        """
        s = StartPool[np.random.randint(0, StartPool.shape[0])]
        indexs = np.where(s != 0)[0]
        noise = np.random.rand(len(indexs)) * A_Scale
        noise = noise - noise.mean()
        return self.step(s, noise)
        
    def judege_s(self, s: np.ndarray) -> bool:
        """
        Judge the state vector.
        
        Parameters:
            s (np.ndarray): state vector
        
        Returns:
            bool: True if the state vector is valid, False otherwise
        """
        if abs(sum(s) - 100) > 0.1 or s.min() < 0 or s.max() > 100:
            return False
        return True
    
    def judge_a(self, s: np.ndarray, a: np.ndarray) -> bool:
        """
        Judge the action vector.
        
        Parameters:
            s (np.ndarray): state vector
            a (np.ndarray): action vector
        
        Returns:
            bool: True if the action vector is valid, False otherwise
        """
        if a.min() < -A_Scale or a.max() > A_Scale:
            return False
        return True

    def reward(self, s, a, s_):
        """
        Calculate the reward based on the performance improvement and threshold achievement.

        Parameters:
            s (np.ndarray): state vector
            a (np.ndarray): action vector
            s_ (np.ndarray): new state vector
        
        Returns:
            float: reward
        """
        result = self.target_func(s)
        result_ = self.target_func(s_)
        # Phase 1 (ensure legal action and state vector)
        if not self.judege_s(s) or not self.judge_a(s, a):
            return -1

        bmg_ = BMGs(s_, result_)
        base_matrix = bmg_.get_base_matrix()

        if bmg_.bmg_s in self.exist_bmgs:
            return 0

        # Calculate the reward based on performance improvement and threshold achievement
        reward = 0
        thresholds = self.matrix_thresholds.get(base_matrix, self.matrix_thresholds['New'])
        for target, weight in RewardWeight.items():
            if weight > 0:
                current_value = result_.get(target, 0)
                previous_value = result.get(target, 0)
                threshold = thresholds[target]

                # Calculate improvement ratio
                improvement_ratio = (current_value - previous_value) / threshold

                # Check if the current value meets or exceeds the threshold
                meets_threshold = current_value >= threshold

                # Compute component of reward
                if meets_threshold and improvement_ratio >= 0:
                    reward += weight * improvement_ratio * 2  # Double the reward if the threshold is met
                else:
                    reward += weight * improvement_ratio
        return reward
                


def unit_test():
    env = Enviroment()
    s = env.reset_random_state()
    indexs = np.where(s != 0)[0]
    for i in range(10):
        a = np.random.rand(len(indexs)) * A_Scale
        a = a - a.mean()
        s_ = env.step(s, a)
        r = env.reward(s, a, s_)
        s = s_
        print(r)

# python -m env.enviroment
if __name__ == '__main__':
    unit_test()
