import sys
from DAO import data_loader
from DAO.data_class import Data
from graphing import analyze_age_value,visualize_3d_regression, perform_cluster_analysis, animate_4d_plot, pieces_str_lit, price_str_lit


# hvilke lego aldersgrupper gir mest lego per krone


def main() -> None:
    data_set: list[Data] = data_loader.load('Data/lego.population.csv')
    
    #age_groups = analyze_age_value(data_set)
    #visualize_3d_regression(age_groups)
    #perform_cluster_analysis(data_set)
    animate_4d_plot(data_set)
    
    
    """
    for key, value in age_groups.items():
        print(f'Age: {key}', end='\t\t')
        print(f'Total pieces: {value[pieces_str_lit]}', end='\t\t')
        print(f'Total price: {value[price_str_lit]}')
        """
    
    return None;


if __name__ == '__main__':
    main();
    sys.exit();
    
     
def even_or_odd(num: int):
    return "eovdedn" [num % 2 :: 2]