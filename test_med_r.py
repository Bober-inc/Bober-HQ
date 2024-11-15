import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def standardize_age(age_str):
    if pd.isna(age_str) or str(age_str).lower() in ['na', 'nan', '']:
        return '3+'
    
    age_str = str(age_str).replace('Ages_', '').lower()
    
    try:
        if '½' in age_str:
            age_str = age_str.replace('½', '.5')
        
        if '-' in age_str:
            start_age = float(age_str.split('-')[0])
        elif '+' in age_str:
            start_age = float(age_str.replace('+', ''))
        else:
            start_age = float(age_str)
        
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
            
    except (ValueError, TypeError):
        return '3+'

# Last og preparer data
df = pd.read_csv('DATA/lego.population.csv', encoding='latin1')
df['Age_Group'] = df['Ages'].apply(standardize_age)

# Konverter priskolonner til numeriske verdier
df['Price'] = pd.to_numeric(df['Price'].str.replace('$', '').str.replace(',', ''), errors='coerce')
df['Amazon_Price'] = pd.to_numeric(df['Amazon_Price'].str.replace('$', '').str.replace(',', ''), errors='coerce')
df['Final_Price'] = df['Price'].fillna(df['Amazon_Price'])
df['Pieces'] = pd.to_numeric(df['Pieces'], errors='coerce')

print("\nFordeling av aldersgrupper:")
print(df['Age_Group'].value_counts().sort_index())

# Beregn regresjoner for hver aldersgruppe
print("\nRegresjonsmetrikker per aldersgruppe:")
for age_group in sorted(df['Age_Group'].unique()):
    group_data = df[df['Age_Group'] == age_group]
    
    # Fjern missing verdier
    mask = group_data['Pieces'].notna() & group_data['Final_Price'].notna()
    X = group_data.loc[mask, 'Pieces'].values.reshape(-1, 1)
    y = group_data.loc[mask, 'Final_Price'].values
    
    if len(X) > 1:  # Sjekk om vi har nok data
        model = LinearRegression()
        model.fit(X, y)
        
        # Beregn prediksjoner
        y_pred = model.predict(X)
        
        # Beregn metrikker
        r2 = r2_score(y, y_pred)
        adj_r2 = 1 - (1 - r2) * (len(y) - 1) / (len(y) - 1 - 1)
        r = np.sqrt(r2)
        
        print(f"\nAldersgruppe: {age_group}")
        print(f"Antall observasjoner: {len(X)}")
        print(f"R²: {r2:.4f}")
        print(f"Justert R²: {adj_r2:.4f}")
        print(f"R (korrelasjonskoeffisient): {r:.4f}")
        print(f"Stigningstall (pris per brikke): ${model.coef_[0]:.3f}")
        print(f"Skjæringspunkt: ${model.intercept_:.2f}")

# Beregn total regresjon
mask_total = df['Pieces'].notna() & df['Final_Price'].notna()
X_total = df.loc[mask_total, 'Pieces'].values.reshape(-1, 1)
y_total = df.loc[mask_total, 'Final_Price'].values

model_total = LinearRegression()
model_total.fit(X_total, y_total)
y_pred_total = model_total.predict(X_total)

r2_total = r2_score(y_total, y_pred_total)
adj_r2_total = 1 - (1 - r2_total) * (len(y_total) - 1) / (len(y_total) - 1 - 1)
r_total = np.sqrt(r2_total)

print("\nTotal regresjon (alle aldersgrupper):")
print(f"Antall observasjoner: {len(X_total)}")
print(f"R²: {r2_total:.4f}")
print(f"Justert R²: {adj_r2_total:.4f}")
print(f"R (korrelasjonskoeffisient): {r_total:.4f}")
print(f"Stigningstall (pris per brikke): ${model_total.coef_[0]:.3f}")
print(f"Skjæringspunkt: ${model_total.intercept_:.2f}")

# Visualiser regresjonene
plt.figure(figsize=(12, 8))
colors = sns.color_palette("husl", n_colors=len(df['Age_Group'].unique()))

for age_group, color in zip(sorted(df['Age_Group'].unique()), colors):
    mask = (df['Age_Group'] == age_group) & df['Pieces'].notna() & df['Final_Price'].notna()
    plt.scatter(df[mask]['Pieces'], df[mask]['Final_Price'], 
               alpha=0.5, label=f'{age_group}', color=color)
    
    # Legg til regresjonslinje for hver gruppe
    if sum(mask) > 1:
        X_group = df[mask]['Pieces'].values.reshape(-1, 1)
        y_group = df[mask]['Final_Price'].values
        
        if len(X_group) > 1:
            model_group = LinearRegression()
            model_group.fit(X_group, y_group)
            X_sort = np.sort(X_group, axis=0)
            y_pred = model_group.predict(X_sort)
            plt.plot(X_sort, y_pred, '--', color=color, alpha=0.8)

plt.xlabel('Antall brikker')
plt.ylabel('Pris ($)')
plt.title('Pris vs. Antall brikker med regresjonslinjer per aldersgruppe')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()