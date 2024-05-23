import math
import random
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from BMGs import BMGs
from config import *
from env.ML_model import ML_Model


class Enviroment:
    def __init__(self) -> None:
        self.n_actions = N_Action
        self.n_states = N_State
        # Models
        self.models = ML_Model(DataPath, DropColumns, TargetColumns, MLResultPath)
        self.best_models = self.models.get_best_models()
        # Thresholds
        self.matrix_thresholds, self.exist_bmgs = self.define_matrix_thresholds(DataPath, Percentile)
        # Record new BMGs
        self.new_bmgs = []
        # Initial Pool and constraints
        self.init_data = pd.read_excel(DataPath).drop(columns=DropColumns).drop(columns=TargetColumns)
        self.init_pool = self.init_data.values
        self.max_com_num = min(self.init_data.astype(bool).sum(axis=1).max(), N_Action)
        self.min_com_num = max(self.init_data.astype(bool).sum(axis=1).min(), 3)
    
    @staticmethod
    def define_matrix_thresholds(excel_file: str, percentile: float = 0.8) -> Tuple[Dict, Dict]:
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
        exist_bmgs = {composition: 0 for composition in df['Chemical composition'].unique().tolist()}
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
        X = s.reshape(1, -1)
        results = {}
        for target, model in self.best_models.items():
            results[target] = max(model.predict(X)[0], 0)
        return results
    
    def step(self, s: np.ndarray, a: np.ndarray) -> Tuple[np.ndarray, float, bool]:
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
        assert len(a) >= len(indexs)
        s_[indexs] = s_[indexs] + a[:len(indexs)]
        reward, done = self.reward(s, a, s_)
        return s_, reward, done
    
    
    def get_random_legal_action(self, s: np.ndarray) -> np.ndarray:
        """
        Get a random legal action vector.
        
        Parameters:
            s (np.ndarray): state vector
        
        Returns:
            np.ndarray: action vector
        """
        iter = 1
        k = np.count_nonzero(s)
        while True:
            a = np.zeros(N_Action)
            action = np.random.rand(k)
            action = (action - action.mean()) * A_Scale
            a[:k] = action
            s_ = self.step(s, a)[0]
            if self.judge_a(a) and self.judge_s(s_):
                break
            iter += 1
        return a
    
    def reset(self):
        """
        Reset the state vector randomly. (In the initial pool)
        
        Returns:
            np.ndarray: state vector
        """
        s = self.init_pool[np.random.randint(0, self.init_pool.shape[0])]
        indexs = np.where(s != 0)[0]
        if len(indexs) < self.min_com_num or len(indexs) > self.max_com_num:
            s = self.reset() # Reset if the number of elements is out of range
        # 50%的概率使用OptionalRestElement替换s中成分最小的元素，保证s中元素个数不变，同时也要保证OptionalRestElement中选中的元素不在s中
        if np.random.rand() > 0.5:
            s = self.replace_element(s)
        return s
    
    def replace_element(self, s: np.ndarray) -> np.ndarray:
        """
        Replace the element with the smallest value in the state vector with an element from the optional rest elements.
        
        Parameters:
            s (np.ndarray): state vector
        
        Returns:
            np.ndarray: state vector
        """
        indexs = np.where(s != 0)[0]
        min_index = np.argmin(s[indexs])
        temp_value = s[indexs[min_index]]
        s[indexs[min_index]] = 0
        # TypeError: only integer scalar arrays can be converted to a scalar index
        rest_elements = OptionalResetElement - set(np.array(CompositionColumns)[indexs])
        new_element = random.choice(list(rest_elements))
        s[CompositionColumns.index(new_element)] = temp_value
        return s
    
    
    def reset_by_constraint(self, mandatory_elements: Dict[str, Tuple[int]], optional_elements: Dict[str, Tuple[int]], k: int) -> np.ndarray:
        """
        Reset the state vector based on the constraints.
        
        Parameters:
            mandatory_elements (dict): mandatory elements with range
                {
                    'Zr': (40, 70),
                    'Cu': (10, 25),
                    'Ni': (5, 15),
                    'Al': (5, 15)
                }
            optional_elements (dict): optional elements with range
                {
                    'Ag': (0, 10),
                    'Ti': (0, 10),
                    'La': (0, 10),
                    'Ce': (0, 10),
                    'Gd': (0, 10),
                    'Y': (0, 10)
                }
            k (int): number of optional elements with range
        
        Returns:
            np.ndarray: state vector
        """
        optional_len = k - len(mandatory_elements)
        
        # Step 1: Randomly select optional elements
        selected_optional_elements = random.sample(list(optional_elements.keys()), optional_len)
        
        # Step 2: Generate random values for mandatory and selected optional elements
        state = {}
        
        for element, (low, high) in mandatory_elements.items():
            state[element] = np.round(np.random.uniform(low, high), 2)
        
        for element in selected_optional_elements:
            low, high = optional_elements[element]
            state[element] = np.round(np.random.uniform(low, high), 2)
        
        # Step 3: Normalize the values to sum to 100
        total = sum(state.values())
        for element in state:
            state[element] = np.round((state[element] / total) * 100, 2)
        
        # Step 4: Create the final state vector
        state_vector = np.zeros(len(CompositionColumns))
        for i, element in enumerate(CompositionColumns):
            if element in state:
                state_vector[i] = state[element]
        
        return state_vector
        
    def judge_s(self, s: np.ndarray) -> bool:
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
    
    def judge_a(self, a: np.ndarray) -> bool:
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

    def reward(self, s: np.ndarray, a: np.ndarray, s_: np.ndarray, result: Optional[Dict[str, float]]= None, result_: Optional[Dict[str, float]]= None) -> Tuple[float, bool]:
        """
        Calculate the reward based on the performance improvement and threshold achievement.

        Parameters:
            s (np.ndarray): state vector
            a (np.ndarray): action vector
            s_ (np.ndarray): new state vector
        
        Returns:
            float: reward
            bool: done
        """
        if not result:
            result = self.target_func(s)
        if not result_:
            result_ = self.target_func(s_)
        # Phase 1 (ensure legal action and state vector)
        bmg_ = BMGs(s_, result_)
        base_matrix_ = bmg_.get_base_matrix()
        
        bmg = BMGs(s, result)
        base_matrix = bmg.get_base_matrix()
        
        if not self.judge_s(s_) or not self.judge_a(a) or base_matrix != base_matrix_:  # Illegal action or state vector
            reward = -10
            return reward, True

        # Calculate the reward based on performance improvement and threshold achievement
        reward = 0
        thresholds = self.matrix_thresholds.get(base_matrix, self.matrix_thresholds['New'])
        thresholds['Dmax(mm)'] = min(5, thresholds['Dmax(mm)']) # Limit the maximum value of Dmax(mm) to 5
        done_count, weight_count = 0, 0
        for target, weight in RewardWeight.items():
            if weight > 0:
                weight_count += 1
                current_value = result_.get(target, 0)
                previous_value = result.get(target, 0)
                threshold = thresholds[target]
                # Calculate improvement ratio
                improvement_ratio = (current_value - previous_value) / threshold

                # Check if the current value meets or exceeds the threshold
                meets_threshold = current_value >= threshold
                
                if current_value >= threshold * DoneRatio:
                    done_count += 1
                # Compute component of reward
                if meets_threshold:
                    reward += 5 # Reward for meeting the threshold
                    reward += weight * improvement_ratio * 2  # Double the reward if the threshold is met
                else:
                    reward += weight * improvement_ratio
        if done_count == weight_count:
            done = True
        else:
            done = False
        if done:
            if self.exist_bmgs.get(bmg_.bmg_s, 0) == 0:
                self.exist_bmgs[bmg_.bmg_s] = 1
                reward += 10
                result_['BMGs'] = bmg_.bmg_s
                self.new_bmgs.append(result_)
            else:
                self.exist_bmgs[bmg_.bmg_s] += 1
                reward += Alpha * math.sqrt((2 * math.log(MaxStep))/self.exist_bmgs[bmg_.bmg_s])

        return reward, done
                


def unit_test():
    env = Enviroment()
    mandatory_elements = {
    'Zr': (40, 70),
    'Cu': (10, 25),
    'Ni': (5, 15),
    'Al': (5, 15)
    }
    optional_elements = {
        'Ag': (0, 10),
        'Ti': (0, 10),
        'La': (0, 10),
        'Ce': (0, 10),
        'Gd': (0, 10),
        'Y': (0, 10)
    } 
    for epoch in range(100):
        # s = env.reset_by_constraint(mandatory_elements, optional_elements, 5)
        s = env.reset()
        for step in range(MaxStep):
            indexs = np.where(s != 0)[0]
            a = np.random.rand(len(indexs)) * A_Scale
            a = a - a.mean()
            s_, r, done = env.step(s, a)
            if done:
                print(r, done)
                break
            s = s_

# python -m env.enviroment
if __name__ == '__main__':
    unit_test()
