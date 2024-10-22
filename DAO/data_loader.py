import pandas as pd
from data_class import Data


def load(filename: str) -> None:
    
    obj = [Data]
    
    
    
    df = pd.read_csv(filename, encoding='ISO-8859-1')
    
    headers = df.columns
    """
    ['Item_Number', 'Set_Name', 'Theme', 'Pieces', 'Price', 'Amazon_Price',
           'Year', 'Ages', 'Pages', 'Minifigures', 'Packaging', 'Weight',
           'Unique_Pieces', 'Availability', 'Size']
    """
    
    for _, row in df.iterrows():
        obj.append(Data(row[1]))
    
    print(headers)
