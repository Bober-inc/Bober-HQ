import sys
from DAO import data_loader


# hvilke lego aldersgrupepr gir flest lego per krone


def main() -> None:
    data_loader.load('Data/lego.population.csv')
    print("Bober")
    return None;


if __name__ == '__main__':
    main();
    sys.exit();
    