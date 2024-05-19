# Data config
DataPath = '../data/ALL_data_grouped_processed.xlsx'  # Replace with your file path
DropColumns = ['BMGs', "Chemical composition"]
TargetColumns = ['Tg(K)', 'Tx(K)', 'Tl(K)', 'Dmax(mm)','yield(MPa)', 'Modulus (GPa)', 'Ε(%)']
CompositionColumns = ['Ni', 'Cr', 'Nb', 'P', 'B', 'Si', 'Fe', 'C', 'Mo', 'Y', 'Co', 'Au', 'Ge', 'Pd', 'Cu', 'Zr', 'Ti', 'Al', 'Mg', 'Ag', 'Gd', 'La', 'Ga', 'Hf', 'Sn', 'In', 'Ca', 'Zn', 'Nd', 'Er', 'Dy', 'Pr', 'Ho', 'Ce', 'Sc', 'Ta', 'Mn', 'Tm', 'Pt', 'V', 'W', 'Tb', 'Li', 'Sm', 'Lu', 'Yb', 'Pb', 'Sr', 'Ru']
MLResultPath = '../results/ML_All'


# PER Config
TrustPoolPath = '../data/trust_pool.jsonl'
PoolSize = 50000

# Reward cnofig
RewardWeight = {
    'Tg(K)': 0,
    'Tx(K)': 0,
    'Tl(K)': 0,
    'yield(MPa)': 0.5,
    'Modulus (GPa)': 0.0,
    'Ε(%)': 0.5
}