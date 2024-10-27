import pandas as pd
from DAO.data_class import Data


def load(filename: str) -> list[Data]:        
    return [Data(item) for item in pd.read_csv(filename, encoding='ISO-8859-1').values]
    