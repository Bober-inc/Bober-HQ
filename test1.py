import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Last og preparer data
df = pd.read_csv('DATA/lego.population.csv', encoding='latin1')

# Konverter pris og fjern '$' tegn
df['Price'] = df['Price'].str.replace('$', '').str.replace(',', '').astype(float)
df['Pieces'] = pd.to_numeric(df['Pieces'])

# Standardiser aldersgrupper
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

df['Age_Group'] = df['Ages'].apply(standardize_age)

# Analyser hver aldersgruppe
print("\nAnalyse per aldersgruppe:")
for age in sorted(df['Age_Group'].unique()):
    group_data = df[df['Age_Group'] == age].dropna(subset=['Pieces', 'Price'])
    if len(group_data) > 1:
        X = group_data['Pieces'].values.reshape(-1, 1)
        y = group_data['Price'].values
        
        model = LinearRegression()
        model.fit(X, y)
        r2 = r2_score(y, model.predict(X))
        
        print(f"\nAldersgruppe: {age}")
        print(f"Antall sett: {len(group_data)}")
        print(f"Intercept: ${model.intercept_:.2f}")
        print(f"Pris per brikke: ${model.coef_[0]:.3f}")
        print(f"R²: {r2:.3f}")

# Total regresjon
print("\nTotal regresjon (alle aldersgrupper):")
X_total = df['Pieces'].values.reshape(-1, 1)
y_total = df['Price'].values
mask = ~np.isnan(X_total.flatten()) & ~np.isnan(y_total)
X_total = X_total[mask]
y_total = y_total[mask]

model_total = LinearRegression()
model_total.fit(X_total, y_total)
r2_total = r2_score(y_total, model_total.predict(X_total))

print(f"Intercept: ${model_total.intercept_:.2f}")
print(f"Pris per brikke: ${model_total.coef_[0]:.3f}")
print(f"R² total: {r2_total:.3f}")

# Print fordeling av sett
print("\nFordeling av sett per aldersgruppe:")
print(df['Age_Group'].value_counts().sort_index())