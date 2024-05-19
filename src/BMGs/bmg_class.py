import pandas as pd
from config import DataPath, CompositionColumns
from typing import Dict
import numpy as np

class BMGs():
    def __init__(self, s: np.ndarray, properties: Dict = None) -> None:
        """
        Initialize the BMGs class with a 56-dimensional vector 's' and a datapath to an Excel file.
        
        Parameters:
            s (list or array-like): vector
        """
        self.s = s
        self.properties = properties
        df = pd.read_excel(DataPath)
        self.bmg_s = self._generate_BMGs_string()
        
    def _generate_BMGs_string(self):
        """
        Generate a BMGs string based on the vector 's' and the columns of the Excel file.
        
        Returns:
            str: BMGs string
        """
        
        columns = CompositionColumns
        assert len(self.s) == len(columns), "The length of the vector 's' must match the number of columns in the Excel file."

        # Create a dictionary from relevant columns and vector 's'
        col_s_dict = {col: self.s[i] for i, col in enumerate(columns)}
        
        # Sort columns by the value in 's' and create the BMGs string
        sorted_cols = sorted(col_s_dict, key=lambda c: col_s_dict[c], reverse=True)
        bmg_string = ''.join([f"{col}{col_s_dict[col]}" for col in sorted_cols if col_s_dict[col] != 0])
        
        return bmg_string
    
    def get_base_matrix(self):
        """
        Get the base matrix of the BMGs.
        
        Returns:
            str: base matrix
        """
        return CompositionColumns[np.argmax(self.s)]
    
    def __str__(self) -> str:
        return f'BMGs: {self.bmg_s}' + '\n' + f'Properties: {self.properties}'
    
def unit_test():
    s = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 15.4, 0.0, 0.0, 0.0, 23.1, 30.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 30.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    bmg = BMGs(s)
    print(bmg)
    print(bmg.get_base_matrix())

# python -m BMGs.bmg_class 
if __name__ == '__main__':
    unit_test()