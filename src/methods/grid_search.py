from env.environment import Environment
import numpy as np
import pandas as pd 

# python -m methods.grid_search
if __name__ == '__main__':
    data_path = '../data/ALL_data_grouped_processed.xlsx' 
    drop_columns = ['BMGs', "Chemical composition", 'cls_label']
    target_columns = ['Tg(K)', 'Tx(K)', 'Tl(K)', 'Dmax(mm)', 'yield(MPa)', 'Modulus (GPa)', 'Ε(%)']
    results_path = '../results/ML_All'
    env = Environment()
    best_models = env.best_models
    compositions = pd.read_excel(data_path).drop(columns = drop_columns).drop(columns = target_columns).columns
    random_search_data_path = './methods/random_search.xlsx'
    random_data = pd.read_excel(random_search_data_path)
    result_df = pd.DataFrame()
    for target_column in target_columns:
        print(target_column, 'predicting...')
        X = random_data[compositions].to_numpy()
        X_sums = X.sum(axis=1, keepdims=True)
        X = X / X_sums
        pred_Y = best_models[target_column].predict(X)
        result_df[target_column] = env.inverse_normal_targets(target_column, pred_Y)
    # 横向拼接
    result_df = pd.concat([random_data, result_df], axis=1) 
    result_df['BMGs'] = result_df.apply(lambda x: ''.join([c + str(x[c]) for c in sorted(x.index[:-1], key=lambda c: x[c], reverse=True) if x[c]!=0 and c not in target_columns and c not in drop_columns]), axis=1)
    BMGs = pd.read_excel(data_path)['BMGs'].tolist()
    # 丢弃已经在BMGs中的数据
    result_df = result_df[~result_df['BMGs'].isin(BMGs)]
    result_df.to_excel('./random_search_result.xlsx', index=False)
    result_df = pd.read_excel('./random_search_result.xlsx')
    # 筛选 Dmax(mm): 5-20 ; yield(MPa): 1500-2000; E(%):10-35%的数据
    result_df = result_df[(result_df['Dmax(mm)'] >= 5) & (result_df['Dmax(mm)'] <= 20) & (result_df['yield(MPa)'] >= 1500) & (result_df['yield(MPa)'] <= 2500) & (result_df['Ε(%)'] >= 9) & (result_df['Ε(%)'] <= 25)]
    result_df.to_excel('./random_search_result_filtered.xlsx', index=False)
    # 找到每一行result_df 中compositions列与data_path的compositions列之差的sum值最小的3行的index,组成一个数组放到result_df的新列 similar_index中,注意我要找3行
    # 循环result_df中的每一行   
    original_data = pd.read_excel(data_path)
    original_features = original_data.drop(columns = drop_columns).drop(columns = target_columns)
    for index, row in result_df.iterrows():
        # 计算差值
        diff = row[compositions].to_numpy() - original_features.to_numpy()
        diff = np.sum(np.abs(diff), axis=1)
        # 找到最小的3个index
        similar_index = np.argsort(diff)[:3]
        # 将这3个index根据target_columns的列名和值 还有BMGs列的值拼接成一个字符串
        similar_BMGs = [original_data.loc[i, 'BMGs'] + ' ' + ' '.join([f'{column}:{original_data.loc[i, column]}' for column in target_columns]) for i in similar_index]
        result_df.loc[index, 'similar_index'] = '\n'.join(similar_BMGs)
    result_df.to_excel('./random_search_result_filtered_similar.xlsx', index=False)