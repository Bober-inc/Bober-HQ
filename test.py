import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression

def load_and_clean_data(path):
    # Last inn data
    df = pd.read_csv(path, sep=',', encoding='latin1')
    
    # Konverter pris- og antallskolonner til numeriske verdier
    df['Price'] = pd.to_numeric(df['Price'].str.replace('$', ''), errors='coerce')
    df['Amazon_Price'] = pd.to_numeric(df['Amazon_Price'].str.replace('$', ''), errors='coerce')
    df['Pieces'] = pd.to_numeric(df['Pieces'], errors='coerce')
    
    # Forbedret alderskonvertering
    def standardize_age(age_str):
        if pd.isna(age_str) or str(age_str).lower() in ['na', 'nana', '']:
            return '3+'  # Default for manglende verdier
            
        # Fjern 'Ages_' og konverter til lowercase
        age_str = str(age_str).replace('Ages_', '').lower()
        
        # Håndter spesialtilfeller
        if '½' in age_str:
            # Konverter 1½ til 1.5
            age_str = age_str.replace('½', '.5')
        
        try:
            # For intervaller (f.eks. "7-14")
            if '-' in age_str:
                start_age = float(age_str.split('-')[0])
            # For enkle aldere med '+' (f.eks. "6+")
            elif '+' in age_str:
                start_age = float(age_str.replace('+', ''))
            # For enkle aldere uten '+' 
            else:
                start_age = float(age_str)
            
            # Konverter float til int for kategorisering
            start_age = int(start_age)
            
            # PEGI kategorisering
            if start_age <= 3:
                return '3+'
            elif start_age <= 7:
                return '7+'
            elif start_age <= 12:
                return '12+'
            elif start_age <= 16:
                return '16+'
            else:
                return '18+'
                
        except ValueError as e:
            #print(f"Kunne ikke konvertere alder: {age_str}, Error: {e}")
            return '3+'  # Default verdi
    
    # Anvendt databehandling
    df['Age_Group'] = df['Ages'].apply(standardize_age)
    df['Final_Price'] = df['Price'].fillna(df['Amazon_Price'])
    df['Price_Per_Piece'] = df['Final_Price'] / df['Pieces']
    
    # Print debugging info
    print("\nEksempler på alderskonvertering:")
    sample_ages = df['Ages'].sample(n=5)
    for age in sample_ages:
        print(f"Original: {age} -> Konvertert: {standardize_age(age)}")
    
    print("\nUnike aldersgrupper etter konvertering:")
    print(df['Age_Group'].value_counts().sort_index())
    
    # Fjern rader med manglende eller ugyldige verdier
    df = df.dropna(subset=['Pieces', 'Final_Price', 'Age_Group'])
    
    return df
    

def calculate_regression_metrics(X, y):
    """Beregn R², justert R² og R for en regresjon"""
    n = len(y)
    p = 1  # antall prediktorer (1 for enkel regresjon)
    
    r2 = r2_score(y, X)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    r = np.sqrt(r2)
    
    return r2, adj_r2, r


# Last og analyser data
try:
    df = load_and_clean_data('DATA/lego.population.csv')
    
    print("\nDataset statistikk:")
    print("\nGjennomsnittlig pris per brikke for hver aldersgruppe:")
    print(df.groupby('Age_Group')['Price_Per_Piece'].mean().round(4))
    
    print("\nAntall sett per aldersgruppe:")
    print(df['Age_Group'].value_counts().sort_index())
    
    # Plott
    plt.figure(figsize=(15, 10))
    
    # Boksplott
    plt.subplot(2, 2, 1)
    sns.boxplot(x='Age_Group', y='Price_Per_Piece', data=df)
    plt.title("Pris per brikke fordelt på aldersgrupper")
    plt.xlabel("Aldersgruppe")
    plt.ylabel("Pris per brikke ($)")
    plt.xticks(rotation=45)
    
    # Scatter plot
    plt.subplot(2, 2, 2)
    for age_group in sorted(df['Age_Group'].unique()):
        mask = df['Age_Group'] == age_group
        plt.scatter(df[mask]['Pieces'], df[mask]['Final_Price'], 
                   alpha=0.5, label=age_group)
    plt.xlabel("Antall brikker")
    plt.ylabel("Pris ($)")
    plt.legend()
    plt.title("Pris vs. Antall brikker per aldersgruppe")
    
    # Histogram av pris per brikke
    plt.subplot(2, 2, 3)
    sns.histplot(data=df, x='Price_Per_Piece', hue='Age_Group', multiple="stack")
    plt.title("Fordeling av pris per brikke")
    plt.xlabel("Pris per brikke ($)")
    
    # Gjennomsnittlig antall brikker per aldersgruppe
    plt.subplot(2, 2, 4)
    df.groupby('Age_Group')['Pieces'].mean().plot(kind='bar')
    plt.title("Gjennomsnittlig antall brikker per aldersgruppe")
    plt.xlabel("Aldersgruppe")
    plt.ylabel("Antall brikker")
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Statistisk analyse
    print("\nDetaljert statistikk per aldersgruppe:")
    stats_summary = df.groupby('Age_Group').agg({
        'Price_Per_Piece': ['count', 'mean', 'std', 'median'],
        'Pieces': ['mean', 'std'],
        'Final_Price': ['mean', 'std']
    }).round(3)
    print(stats_summary)
    
    # ANOVA test
    age_groups = [group['Price_Per_Piece'].values for name, group in df.groupby('Age_Group')]
    if len(age_groups) > 1:
        f_stat, p_val = stats.f_oneway(*age_groups)
        print("\nANOVA test resultater:")
        print(f"F-statistikk: {f_stat:.2f}")
        print(f"p-verdi: {p_val:.4f}")

except Exception as e:
    print(f"En feil oppstod: {str(e)}")
    import traceback
    traceback.print_exc()