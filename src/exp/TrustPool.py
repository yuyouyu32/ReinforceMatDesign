import json
import os
from itertools import combinations

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import *
from env.enviroment import Enviroment


class TrustPool:
    def __init__(self):
        
        self.data = pd.read_excel(DataPath).drop(columns=DropColumns)
        self.env = Enviroment()
        self.models = self.env.models
        self.fill_na_with_models()


    def fill_na_with_models(self):
        best_models = self.models.get_best_models()
        for target_column in self.models.target_columns:
            X_pred = self.data[self.data[target_column].isna()].drop(columns=self.models.target_columns)
            if X_pred.empty:
                continue

            y_preds = best_models[target_column].predict(X_pred.to_numpy())
            
            min_value = self.data[target_column].min()
            max_value = self.data[target_column].max()
            # adjust the predicted values to be within the range of the original data(fix bug with edRVFL predictions)
            y_preds_adjusted = [max(min(y, max_value * 1.25), min_value * 0.75) for y in y_preds]
            
            self.data.loc[self.data[target_column].isna(), target_column] = y_preds_adjusted


    def generate_experience_pool(self, threshold, jsonl_path, rewrite=False):
        trust_exp_pool = []
        if os.path.exists(jsonl_path):
            if not rewrite:
            # return all the data in the jsonl file
                with open(jsonl_path, 'r') as f:
                    for line in f:
                        trust_exp_pool.append(json.loads(line))
                return trust_exp_pool
            else:
                os.remove(jsonl_path)
        # if the file does not exist or rewrite is True, generate the data and write it to the file

        data_features = self.data.drop(columns=self.models.target_columns).values

        for i, j in tqdm(combinations(range(len(data_features)), 2), total=len(data_features) * (len(data_features) - 1) // 2):
            row_i = data_features[i]
            row_j = data_features[j]

            diff = row_i - row_j
            abs_diff = np.abs(diff)

            if np.all(abs_diff <= threshold):
                s, s_ , a = row_i.tolist(), row_j.tolist(), diff.tolist()
                r, done = self.env.reward(np.array(s), np.array(a), np.array(s_))
                
                diff_map = {
                    's': s.copy(),
                    's_': s_.copy(),
                    'a': a.copy(),
                    'r': r,
                    'done': done
                }

                for col in self.models.target_columns:
                    diff_map[col] = self.data.at[i, col]
                    diff_map[f"{col}_"] = self.data.at[j, col]

                # Record the reverse experience
                s, s_ , a = row_j.tolist(), row_i.tolist(), (-diff).tolist()
                r, done = self.env.reward(np.array(s), np.array(a), np.array(s_))
                diff_map_reverse = {
                    's': s.copy(),
                    's_': s_.copy(),
                    'a': a.copy(),
                    'r': r,
                    'done': done
                }

                for col in self.models.target_columns:
                    diff_map_reverse[col] = self.data.at[j, col]
                    diff_map_reverse[f"{col}_"] = self.data.at[i, col]

                trust_exp_pool.append(diff_map)
                trust_exp_pool.append(diff_map_reverse)
                with open(jsonl_path, 'a') as f:
                    f.write(json.dumps(diff_map, ensure_ascii=False) + '\n')
                    f.write(json.dumps(diff_map_reverse, ensure_ascii=False) + '\n')
                    



def unit_test():
    trust_pool = TrustPool()
    # trust_pool.data.describe().to_excel('/Users/yuyouyu/WorkSpace/Mine/ReinforceMatDesign/data/ALL_data_grouped_processed_filled_des.xlsx')
    trust_pool.generate_experience_pool(A_Scale, '/Users/yuyouyu/WorkSpace/Mine/ReinforceMatDesign/data/trust_pool.jsonl', rewrite=True)

# python -m exp.TrustPool
if __name__ == '__main__':
    unit_test()