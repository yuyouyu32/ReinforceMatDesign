from env.ML_model import *



if __name__ == '__main__':
    data_path = '/Users/yuyouyu/WorkSpace/Mine/ReinforceMatDesign/data/ALL_data_grouped_processed.xlsx' 
    drop_columns = ['BMGs', "Chemical composition"]
    target_columns = ['Tg(K)', 'Tx(K)', 'Tl(K)', 'Dmax(mm)', 'yield(MPa)', 'Modulus (GPa)', 'Ε(%)']
    results_path = '/Users/yuyouyu/WorkSpace/Mine/ReinforceMatDesign/results/ML_All'

    ml_model = ML_Model(data_path, drop_columns, target_columns, results_path)
    best_models = ml_model.get_best_models()
    compositions = pd.read_excel(data_path).drop(columns = drop_columns).drop(columns = target_columns).columns
    random_search_data_path = '/Users/yuyouyu/WorkSpace/Mine/ReinforceMatDesign/data/random_search.xlsx'
    random_data = pd.read_excel(random_search_data_path)
    result_df = pd.DataFrame()
    for target_column in target_columns:
        result_df[target_column] = best_models[target_column].predict(random_data[compositions].to_numpy())
    # 横向拼接
    result_df = pd.concat([random_data, result_df], axis=1) 
    result_df['BMGs'] = result_df.apply(lambda x: ''.join([c + str(int(x[c])) for c in sorted(x.index[:-1], key=lambda c: x[c], reverse=True) if x[c]!=0 and c not in target_columns and c not in drop_columns]), axis=1)
    BMGs = pd.read_excel(data_path)['BMGs'].tolist()
    # 丢弃已经在BMGs中的数据
    result_df = result_df[~result_df['BMGs'].isin(BMGs)]
    result_df.to_excel('/Users/yuyouyu/WorkSpace/Mine/ReinforceMatDesign/results/random_search_result.xlsx')