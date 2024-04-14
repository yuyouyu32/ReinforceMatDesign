import pandas as pd
from config import DataPath, DropColumns, TargetColumns


InitData = pd.read_excel(DataPath).drop(columns=DropColumns).drop(columns=TargetColumns)
CompositionClomuns = InitData.columns.to_list()
StartPool = InitData.values
N_State = len(CompositionClomuns)
N_Action = 7
A_Scale = 10

MaxComNum = min(InitData.astype(bool).sum(axis=1).max(), 6)
MinComNum = max(InitData.astype(bool).sum(axis=1).min(), 4)
