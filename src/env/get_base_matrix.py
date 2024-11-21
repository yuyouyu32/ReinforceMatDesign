import pandas as pd
from config import *
import json

def get_row_matrix(df: pd.DataFrame):
    max_values = df.max(axis=1)
    df['element_base'] = df.apply(
        lambda row: row[row == max_values[row.name]].index.tolist(),
        axis=1
    )
    # df_expanded = df.explode('element_base').reset_index(drop=True)
    return df


def get_base_matrix_dict(df: pd.DataFrame):
    grouped = df.groupby('element_base')
    
    result = {}

    for name, group in grouped:
        always_present = {}
        possibly_present = {}
        
        # 忽略element_base列
        group_data = group.drop(columns=['element_base'])
        
        for column in group_data.columns:
            if column in DropColumns or column in TargetColumns:
                continue
            min_val = group_data[column].min()
            max_val = group_data[column].max()
            if min_val > 0:
                always_present[column] = (float(min_val) * 0.8 / 100, min(float(max_val) * 1.2/ 100, 1))
            elif max_val > 0:
                possibly_present[column] = (0.0, min(float(max_val) * 1.2 / 100, 1))
        
        # 计算每行中元素个数大于0的最大值
        max_elements = int(group_data.apply(lambda row: row.gt(0).sum(), axis=1).max())
        
        result[name] = (always_present, possibly_present, max_elements)
        
    return result

def main():
    data_path = '../data/ALL_data_grouped_processed.xlsx'
    df = pd.read_excel(data_path)
    # 筛选出data_path中cls_label为1的行
    df = df[df['cls_label'] == 1]
    df_f = df.drop(columns=DropColumns).drop(columns=TargetColumns)
    print(len(df))
    df_f = get_row_matrix(df_f)
    df['element_base'] = df_f['element_base']
    df_expanded = df.explode('element_base').reset_index(drop=True)
    df_expanded.to_excel('../data/BMGs_element_base.xlsx', index=False)
    df_expanded = df_expanded.drop(columns=DropColumns).drop(columns=TargetColumns)
    result = get_base_matrix_dict(df_expanded)
    # ['Ag', 'Al', 'Au', 'Ca', 'Ce', 'Co', 'Cu', 'Dy', 'Er', 'Fe', 'Gd', 'Hf', 'Ho', 'La', 'Lu', 'Mg', 'Mo', 'Nb', 'Nd', 'Ni', 'P', 'Pd', 'Pr', 'Pt', 'Sc', 'Sm', 'Sr', 'Ta', 'Tb', 'Ti', 'Tm', 'W', 'Y', 'Yb', 'Zn', 'Zr']
    json.dump(result, open('../data/base_matrix.json', 'w'), indent=4, ensure_ascii=False)
    
# python -m env.get_base_matrix
if __name__ == "__main__":
    main()