import pandas as pd
from dataloader.my_dataloader import CustomDataLoader
from .ML_worker import ModelEvaluatorKFold
import multiprocessing
import os
from config import logging
logger = logging.getLogger(__name__)


output_path = './Output/'  # Replace with your output path
process_method = 'Single'  # 'Single' or 'Multi'
file_path = '/Users/yuyouyu/WorkSpace/Mine/ReinforceMatDesign/data/ALL_data_grouped_processed.xlsx'  # Replace with your file path
drop_columns = ['BMGs', "Chemical composition"]
target_columns = ['Tg(K)', 'Tx(K)', 'Tl(K)', 'Dmax(mm)', 'yield(MPa)', 'Modulus (GPa)', 'Ε(%)']
Save_path = output_path + 'ML_All/'

def process_target(target_name, file_path, drop_columns, Save_path, target_columns):
    dataloader = CustomDataLoader(file_path, drop_columns, target_columns)
    features, target = dataloader.get_features_for_target(target_name)
    logger.info("{:=^80}".format(f" {target_name} Start"))
    evaluator = ModelEvaluatorKFold(n_splits=5)
    evaluation_results = evaluator.evaluate_models(features, target)
    results = pd.DataFrame(evaluation_results)
    logger.info(f"{target_name}:\n", results, "\n")
    target_name = target_name.replace('/', '_')
    try:
        results.to_excel(Save_path + f"{target_name}_ml.xlsx")
    except FileNotFoundError:
        import os
        os.makedirs(Save_path, exist_ok=True)
        results.to_excel(Save_path + f"{target_name}_ml.xlsx")
    logger.info("{:=^80}".format(f" {target_name} Done"))



if __name__ == "__main__":
    if not os.path.exists(Save_path):
        os.makedirs(Save_path)

    if process_method == 'Multi':
        # Multiprocessing
        processes = []
        for target_name in target_columns:
            p = multiprocessing.Process(target=process_target, args=(target_name, file_path, drop_columns, Save_path, target_columns))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()
    else:
        # Single process
        for target_name in target_columns:
            process_target(target_name, file_path, drop_columns, Save_path, target_columns)
