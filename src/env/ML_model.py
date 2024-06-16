import glob

import pandas as pd
from sklearn.ensemble import (AdaBoostRegressor, GradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

from config import logging
from dataloader.my_dataloader import CustomDataLoader
from MLs.edRVFL import EnsembleDeepRVFL

logger = logging.getLogger(__name__)

class ML_Model:
    def __init__(self, data_path, drop_columns, target_columns, results_path):
        self.data = CustomDataLoader(data_path, drop_columns, target_columns)
        self.target_columns = target_columns
        self.results_path = results_path
    
    def get_best_models(self, norm_features=True, norm_target=True):
        xlsx_paths = glob.glob(self.results_path + '/*.xlsx')
        best_models = {}
        for target_column in self.target_columns:
            # Find the best model for each target column
            for xlsx_path in xlsx_paths:
                if target_column in xlsx_path:
                    best_models[target_column] = self._get_best_model(xlsx_path)
                    x, y = self.data.get_features_for_target(target_column)
                    x = x.to_numpy()
                    if norm_features:
                        x_sums = x.sum(axis=1, keepdims=True)
                        x = x / x_sums
                    y = y.to_numpy()
                    if norm_target:
                        y = (y - y.min()) / (y.max() - y.min())
                    best_models[target_column].fit(x, y)
                    logger.info(f"Best model for {target_column} is {best_models[target_column].__class__.__name__}")
        return best_models

    def _get_best_model(self, model_path):
        df = pd.read_excel(model_path, index_col=0).T
        # Find the best model based on the highest R2 score
        model_name = df['r2'].astype(float).idxmax()
        df_dict =  df.to_dict()
        best_param = df_dict['best_params'][model_name]

        if model_name == 'Ridge':
            model = Ridge(**eval(best_param))
        elif model_name == 'Lasso':
            model = Lasso(**eval(best_param))
        elif model_name == 'ElasticNet':
            model = ElasticNet(**eval(best_param))
        elif model_name == 'SVR':
            model = SVR(**eval(best_param))
        elif model_name == 'RandomForestRegressor':
            model = RandomForestRegressor(**eval(best_param))
        elif model_name == 'GradientBoostingRegressor':
            model = GradientBoostingRegressor(**eval(best_param))
        elif model_name == 'AdaBoostRegressor':
            model = AdaBoostRegressor(**eval(best_param))
        elif model_name == 'KNeighborsRegressor':
            model = KNeighborsRegressor(**eval(best_param))
        elif model_name == 'XGBRegressor':
            model = XGBRegressor(**eval(best_param))
        elif model_name == 'edRVFL':
            model = EnsembleDeepRVFL(**eval(best_param))
        else:
            raise ValueError("Unknown model name.")
        
        return model

def unit_test():
    data_path = '/Users/yuyouyu/WorkSpace/Mine/ReinforceMatDesign/data/ALL_data_grouped_processed.xlsx'  # Replace with your file path
    drop_columns = ['BMGs', "Chemical composition"]
    target_columns = ['Tg(K)', 'Tx(K)', 'Tl(K)', 'Dmax(mm)', 'yield(MPa)', 'Modulus (GPa)', 'Ε(%)']
    results_path = '/Users/yuyouyu/WorkSpace/Mine/ReinforceMatDesign/results/ML_All'

    ml_model = ML_Model(data_path, drop_columns, target_columns, results_path)
    best_models = ml_model.get_best_models()

if __name__ == '__main__':
    unit_test()