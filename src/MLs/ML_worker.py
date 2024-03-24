import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV
from .config import *
from sklearn.metrics import r2_score, mean_squared_error
import warnings
import sys, os
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses

import numpy as np

class ModelEvaluatorKFold:
    def __init__(self, n_splits=5):
        self.models = models
        self.params = param_grid
        self.best_models = {}
        self.kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    def evaluate_models(self, features, target):
        results = {}
        features = features.to_numpy()
        target = target.to_numpy()
        for model_name, model in self.models.items():
            print(f"Evaluating {model_name} with K-fold cross-validation...")
            param_grid = self.params[model_name]
            
            grid_search = GridSearchCV(model, param_grid, cv=self.kf, scoring='neg_mean_squared_error',verbose=0, n_jobs=12)
            print("{:=^40}".format(f" {model_name} Grid Search"))
            grid_search.fit(features, target)
            # print(pd.DataFrame(grid_search.cv_results_))
            
            best_model = grid_search.best_estimator_
            self.best_models[model_name] = best_model
            
            best_params = grid_search.best_params_
            model.set_params(**best_params)
            r2, rmse, mape = self.eval_model(model, features, target)
            results[model_name] = {'best_params': best_params, 'rmse': rmse, 'r2': r2, 'mape': mape}
            print(f"{model_name} Best Parameters: {best_params}\n", results[model_name])
            print("{:=^40}".format(f" Best {model_name} Parameters Done"))
            
        return results

    def eval_model(self, model, features, target):
        r2_scores = []
        rmse_scores = []
        mape_scores = []
        for train_index, test_index in self.kf.split(features):
            X_train, X_test = features[train_index], features[test_index]
            y_train, y_test = target[train_index], target[test_index]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2_scores.append(r2_score(y_test, y_pred))
            rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))
            # Handling zero values in target for MAPE calculation
            y_test_without_zeros = np.where(y_test != 0, y_test, 1e-8)
            y_pred_without_zeros = np.where(y_test != 0, y_pred, 1e-8)
            mape = np.mean(np.abs((y_test_without_zeros - y_pred_without_zeros) / y_test_without_zeros)) * 100
            mape_scores.append(mape)
        mean_r2_score = np.mean(r2_scores)
        mean_rmse_score = np.mean(rmse_scores)
        mean_mape_score = np.mean(mape_scores)
        return mean_r2_score, mean_rmse_score, mean_mape_score

def unit_test():
    features = pd.DataFrame(np.random.randn(100, 5), columns=['f1', 'f2', 'f3', 'f4', 'f5'])
    target = pd.Series(np.random.randn(100))
    evaluator = ModelEvaluatorKFold(n_splits=5)
    evaluation_results = evaluator.evaluate_models(features, target)
    for model_name, result in evaluation_results.items():
        print(f"Model: {model_name}")
        print(f"Best Parameters: {result['best_params']}")
        print(f"Best Mean Squared Error: {result['best_mse']}")
        print(f"R2 Score: {result['r2']}")
        print(f"Mean Absolute Percentage Error (MAPE): {result['mape']}\n")

if __name__ == "__main__":
    unit_test()
