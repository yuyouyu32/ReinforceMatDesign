from env.ML_model import *
import numpy as np

if __name__ == '__main__':
    data_path = '../data/ALL_data_grouped_processed.xlsx' 
    drop_columns = ['BMGs', "Chemical composition"]
    target_columns = ['Tg(K)', 'Tx(K)', 'Tl(K)', 'Dmax(mm)', 'yield(MPa)', 'Modulus (GPa)', 'Ε(%)']
    results_path = '../results/ML_All'

    # ml_model = ML_Model(data_path, drop_columns, target_columns, results_path)
    # best_models = ml_model.get_best_models()
    compositions = pd.read_excel(data_path).drop(columns = drop_columns).drop(columns = target_columns).columns
    # random_search_data_path = '../data/random_search.xlsx'
    # random_data = pd.read_excel(random_search_data_path)
    # result_df = pd.DataFrame()
    # for target_column in target_columns:
    #     print(target_column, 'predicting...')
    #     result_df[target_column] = best_models[target_column].predict(random_data[compositions].to_numpy())
    # # 横向拼接
    # result_df = pd.concat([random_data, result_df], axis=1) 
    # result_df['BMGs'] = result_df.apply(lambda x: ''.join([c + str(int(x[c])) for c in sorted(x.index[:-1], key=lambda c: x[c], reverse=True) if x[c]!=0 and c not in target_columns and c not in drop_columns]), axis=1)
    # BMGs = pd.read_excel(data_path)['BMGs'].tolist()
    # # 丢弃已经在BMGs中的数据
    # result_df = result_df[~result_df['BMGs'].isin(BMGs)]
    # result_df.to_excel('../results/random_search_result.xlsx')
    result_df = pd.read_excel('../results/random_search_result.xlsx')
    # 筛选 Dmax(mm): 5-20 ; yield(MPa): 1500-2000; E(%):10-35%的数据
    result_df = result_df[(result_df['Dmax(mm)'] >= 5) & (result_df['Dmax(mm)'] <= 20) & (result_df['yield(MPa)'] >= 1500) & (result_df['yield(MPa)'] <= 2200) & (result_df['Ε(%)'] >= 10) & (result_df['Ε(%)'] <= 35)]
    result_df.to_excel('../results/random_search_result_filtered.xlsx')
    # 找到每一行result_df 中compositions列与data_path的compositions列之差的sum值最小的3行的index,组成一个数组放到result_df的新列 similar_index中,注意我要找3行
    # 循环result_df中的每一行   
    original_data = pd.read_excel(data_path).drop(columns = drop_columns).drop(columns = target_columns)
    for index, row in result_df.iterrows():
        # 计算差值
        diff = row[compositions].to_numpy() - original_data.to_numpy()
        diff = np.sum(np.abs(diff), axis=1)
        # 找到最小的3个index
        similar_index = np.argsort(diff)[:3]
        result_df.loc[index, 'similar_index'] = str(similar_index)
    result_df.to_excel('../results/random_search_result_filtered_similar.xlsx')