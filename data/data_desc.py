import pandas as pd

reg_data_path = "./ALL_data_grouped_processed.xlsx"
reg_data = pd.read_excel(reg_data_path, sheet_name="Sheet1")
# reg_data = reg_data[reg_data["cls_label"] == 1]
reg_data.describe().to_excel("./ALL_data_grouped_processed_des.xlsx")

cls_data_path = "./ALL_data_cls.xlsx"
cls_data = pd.read_excel(cls_data_path, sheet_name="Sheet1")
# 统计Class列的分布, 打印每一个label的数量和他的比例，注意要两列，数量也要，比例也要
cls_count = cls_data["Class"].value_counts()
cls_count_ratio = cls_data["Class"].value_counts(normalize=True)
cls_count_df = pd.DataFrame({"count": cls_count, "ratio": cls_count_ratio})
cls_count_df.to_excel("./ALL_data_cls_des.xlsx")