# Data config
DataPath = '../data/ALL_data_grouped_processed.xlsx'  # Replace with your file path
DropColumns = ['BMGs', "Chemical composition"]
TargetColumns = ['Tg(K)', 'Tx(K)', 'Tl(K)', 'Dmax(mm)','yield(MPa)', 'Modulus (GPa)', 'Ε(%)']
CompositionColumns = ['Ni', 'Cr', 'Nb', 'P', 'B', 'Si', 'Fe', 'C', 'Mo', 'Y', 'Co', 'Au', 'Ge', 'Pd', 'Cu', 'Zr', 'Ti', 'Al', 'Mg', 'Ag', 'Gd', 'La', 'Ga', 'Hf', 'Sn', 'In', 'Ca', 'Zn', 'Nd', 'Er', 'Dy', 'Pr', 'Ho', 'Ce', 'Sc', 'Ta', 'Mn', 'Tm', 'Pt', 'V', 'W', 'Tb', 'Li', 'Sm', 'Lu', 'Yb', 'Pb', 'Sr', 'Ru']
MLResultPath = '../results/ML_All'


# ENV config
N_State = len(CompositionColumns)
N_Action = 7
A_Scale = 5
Percentile = 0.8
DoneRatio = 1.2
MaxStep = 100
Alpha = 0.6 # UBC 1 config
OptionalResetElement = {'Ag', 'Ti', 'La', 'Ce', 'Gd', 'Y'}


# PER Config
TrustPoolPath = '../data/trust_pool.jsonl'
PoolSize = 50000

# Reward cnofig
RewardWeight = {
    'Dmax(mm)': 0.2,
    'Tg(K)': 0,
    'Tx(K)': 0,
    'Tl(K)': 0,
    'yield(MPa)': 0.0,
    'Modulus (GPa)': 0.4,
    'Ε(%)': 0.4
}

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

ExploreBases = [(
    {
        'Zr': (40, 70),
        'Cu': (10, 25),
        'Ni': (5, 15),
        'Al': (5, 15)
    },
    {
        'Ag': (0, 10),
        'Ti': (0, 10),
        'La': (0, 10),
        'Ce': (0, 10),
        'Gd': (0, 10),
        'Y': (0, 10)
    },
    5)]