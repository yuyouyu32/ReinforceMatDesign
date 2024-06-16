import glob

import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from config import logging
from dataloader.my_dataloader import CustomDataLoader
from imblearn.over_sampling import SMOTE

logger = logging.getLogger(__name__)

class Cls_Model:
    def __init__(self, data_path, drop_columns, target_columns, results_path):
        self.data = CustomDataLoader(data_path, drop_columns, target_columns)
        self.target_columns = target_columns
        self.results_path = results_path

    def get_best_models(self, norm_features=True):

        cls_model = self._get_best_model(self.results_path + '/cls_label_ml.xlsx')
        x, y = self.data.get_features_for_target(self.target_columns[0])
        x = x.to_numpy()
        smote = SMOTE(random_state=42)
        x, y = smote.fit_resample(x, y)
        if norm_features:
            x_sums = x.sum(axis=1, keepdims=True)
            x = x / x_sums
        
        cls_model.fit(x, y)
        logger.info(f"Best model for {self.target_columns[0]} is {cls_model.__class__.__name__}")
        return cls_model

    def _get_best_model(self, model_path):
        df = pd.read_excel(model_path, index_col=0).T
        # Find the best model based on the highest R2 score
        model_name = df['auc'].astype(float).idxmax()
        df_dict =  df.to_dict()
        best_param = df_dict['best_params'][model_name]

        if model_name == 'LogisticRegression':
            model = LogisticRegression(**eval(best_param))
        elif model_name == 'SVC':
            model = SVC(**eval(best_param))
        elif model_name == 'RandomForestClassifier':
            model = RandomForestClassifier(**eval(best_param))
        elif model_name == 'GradientBoostingClassifier':
            model = GradientBoostingClassifier(**eval(best_param))
        elif model_name == 'AdaBoostClassifier':
            model = AdaBoostClassifier(**eval(best_param))
        elif model_name == 'KNeighborsClassifier':
            model = KNeighborsClassifier(**eval(best_param))
        elif model_name == 'XGBClassifier':
            model = XGBClassifier(**eval(best_param))
        elif model_name == 'DecisionTreeClassifier':
            model = DecisionTreeClassifier(**eval(best_param))
        elif model_name == 'GaussianNB':
            model = GaussianNB(**eval(best_param))
        elif model_name == 'MultinomialNB':
            model = MultinomialNB(**eval(best_param))
        elif model_name == 'BernoulliNB':
            model = BernoulliNB(**eval(best_param))
        elif model_name == 'QuadraticDiscriminantAnalysis':
            model = QuadraticDiscriminantAnalysis(**eval(best_param))
        elif model_name == 'CatBoostClassifier':
            model = CatBoostClassifier(**eval(best_param))
        else:
            raise ValueError(f"Model {model_name} is not recognized.")

        return model

def unit_test():
    data_path = '/data/home/yeyongyu/SHU/ReinforceMatDesign/data/ALL_data_cls.xlsx'  # Replace with your file path
    drop_columns = ['Class', 'GFA', "Chemical composition"]
    target_columns = ['cls_label']
    results_path = '/data/home/yeyongyu/SHU/ReinforceMatDesign/results/Cls'

    ml_model = Cls_Model(data_path, drop_columns, target_columns, results_path)
    best_models = ml_model.get_best_models()
    # print(best_models.predict_proba(x[10:20]))

if __name__ == '__main__':
    unit_test()