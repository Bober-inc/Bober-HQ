import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm

# Les inn datasettet
df = pd.read_csv("Data/lego.population.csv", sep=",", encoding="latin1")

# Standardiser aldersgrupper med PEGI-kategorier
def standardize_age(age_str):
    if pd.isna(age_str):
        return '3+'
    
    age_str = str(age_str).lower().replace('ages_', '').replace('age_', '')
    
    try:
        if '+' in age_str:
            base_age = int(float(age_str.replace('+', '')))
        elif '-' in age_str:
            start, end = map(lambda x: int(float(x)), age_str.split('-'))
            base_age = start
        else:
            base_age = int(float(age_str))
            
        # PEGI-kategorisering
        if base_age <= 3:
            return '3+'
        elif base_age <= 7:
            return '7+'
        elif base_age <= 12:
            return '12+'
        elif base_age <= 16:
            return '16+'
        else:
            return '18+'
            
    except:
        return '3+'

# Anvend standardisering
df['Age_Group'] = df['Ages'].apply(standardize_age)

# Konverter begge priskolonner til numeriske verdier
df['Price_Numeric'] = pd.to_numeric(df['Price'].str.replace('$', ''), errors='coerce')
df['Amazon_Price_Numeric'] = pd.to_numeric(df['Amazon_Price'].str.replace('$', ''), errors='coerce')

# Bruk Amazon_Price der Price er NA
df['Final_Price'] = df['Price_Numeric'].fillna(df['Amazon_Price_Numeric'])

# Fjern rader med manglende verdier
required_columns = ['Pieces', 'Final_Price', 'Age_Group']
df_clean = df.dropna(subset=required_columns)

# Konverter Pieces til numerisk for å være sikker
df_clean['Pieces'] = pd.to_numeric(df_clean['Pieces'])

# Først, kjør enkel regresjon uten alder
X_simple = sm.add_constant(df_clean['Pieces'])
y = df_clean['Final_Price']

model_simple = sm.OLS(y, X_simple).fit()

print("Enkel regresjonsmodell (basis):")
print("=" * 50)
print(f"R² verdi: {model_simple.rsquared:.3f}")
print(f"Konstant: {model_simple.params[0]:.3f}")
print(f"Brikker: {model_simple.params[1]:.3f}")

# Beregn residualer fra enkel modell
base_predictions = model_simple.predict(X_simple)
base_residuals = y - base_predictions

age_effects = pd.DataFrame({
    'Age_Group': df_clean['Age_Group'],
    'Base_Residual': base_residuals
}).groupby('Age_Group')['Base_Residual'].agg(['mean', 'std', 'count', 'sem'])

# Beregn p-verdier for hver aldersgruppe
age_effects['p_value'] = [
    stats.ttest_1samp(
        base_residuals[df_clean['Age_Group'] == age], 
        0
    )[1]
    for age in age_effects.index
]

print("\nAlderseffekter relativt til basismodell:")
print("=" * 70)
print(f"{'Aldersgruppe':<15} {'Effekt':>10} {'Std.feil':>10} {'P-verdi':>10} {'N':>6}")
print("-" * 70)
for age in sorted(age_effects.index):
    print(f"{age:<15} {age_effects.loc[age, 'mean']:10.2f} "
          f"{age_effects.loc[age, 'sem']:10.2f} "
          f"{age_effects.loc[age, 'p_value']:10.4f} "
          f"{age_effects.loc[age, 'count']:6.0f}")

print(f"Konstant: {model_simple.params[0]:.3f}")

X_simple = sm.add_constant(df_clean['Pieces'])
y = df_clean['Final_Price']
model_simple = sm.OLS(y, X_simple).fit()

print("\nRegresjon uten aldersgrupper:")
print("=" * 50)
print(f"R² verdi: {model_simple.rsquared:.3f}")
print(f"Prisformel: Pris = {model_simple.params[0]:.2f} + {model_simple.params[1]:.2f} × (Brikker)")
print("\nDetaljerte resultater:")
print(f"Intercept: {model_simple.params[0]:.3f} (p-verdi: {model_simple.pvalues[0]:.4f})")
print(f"Brikker koeffisient: {model_simple.params[1]:.3f} (p-verdi: {model_simple.pvalues[1]:.4f})")


# QQ-plot og residualfordeling
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# QQ-plot
stats.probplot(base_residuals, dist="norm", plot=ax1)
ax1.set_title("Q-Q plot av residualer")
ax1.set_xlabel("Forventede normalfordelte kvantiler")
ax1.set_ylabel("Observerte residualer")

# Histogram
sns.histplot(base_residuals, kde=True, ax=ax2)
ax2.set_title("Fordeling av residualer")
ax2.set_xlabel("Residualverdi")
ax2.set_ylabel("Antall observasjoner")

plt.tight_layout()
plt.show()

# Boksplot av residualer per aldersgruppe
plt.figure(figsize=(10, 6))
sns.boxplot(x='Age_Group', y='Base_Residual', data=pd.DataFrame({
    'Age_Group': df_clean['Age_Group'],
    'Base_Residual': base_residuals
}))
plt.title('Fordeling av residualer per aldersgruppe')
plt.xlabel('Aldersgruppe')
plt.ylabel('Residual ($)')
plt.xticks(rotation=45)
plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Vis statistikk per aldersgruppe
summary = df_clean.groupby('Age_Group').agg({
    'Final_Price': ['mean', 'count'],
    'Pieces': 'mean',
    'Pieces': lambda x: np.mean(x/df_clean.loc[x.index, 'Final_Price'])  # Brikker per dollar
}).round(2)
summary.columns = ['Gjennomsnittspris', 'Antall', 'Brikker per dollar']
print("\nStatistikk per aldersgruppe:")
print(summary)

# Vis modellens totale forklaringskraft
print("\nModellstatistikk:")
print(f"R² verdi: {model_simple.rsquared:.3f}")
print(f"Justert R² verdi: {model_simple.rsquared_adj:.3f}")
print(f"F-statistikk: {model_simple.fvalue:.2f}")
print(f"Prob (F-statistikk): {model_simple.f_pvalue:.4f}")


base_results = pd.DataFrame({
    'Estimat': model_simple.params,
    'Std.feil': model_simple.bse,
    'P-verdi': model_simple.pvalues,
    'N': len(df_clean)
}, index=['Intercept', 'Brikker'])

# Kombiner med alderseffektene
age_results = pd.DataFrame({
    'Estimat': age_effects['mean'],
    'Std.feil': age_effects['sem'],
    'P-verdi': age_effects['p_value'],
    'N': age_effects['count']
})

# Kombiner og print resultatene
print("\nKombinerte resultater:")
print("=" * 80)
print(f"{'Variabel':<15} {'Estimat':>10} {'Std.feil':>10} {'P-verdi':>10} {'N':>8}")
print("-" * 80)
# Print basismodell resultater
print(f"{'Intercept':<15} {base_results.loc['Intercept', 'Estimat']:10.3f} "
      f"{base_results.loc['Intercept', 'Std.feil']:10.3f} "
      f"{base_results.loc['Intercept', 'P-verdi']:10.4f} {base_results.loc['Intercept', 'N']:8.0f}")
print(f"{'Brikker':<15} {base_results.loc['Brikker', 'Estimat']:10.3f} "
      f"{base_results.loc['Brikker', 'Std.feil']:10.3f} "
      f"{base_results.loc['Brikker', 'P-verdi']:10.4f} {base_results.loc['Brikker', 'N']:8.0f}")
print("-" * 80)
# Print alderseffekter
for age in sorted(age_results.index):
    print(f"{age:<15} {age_results.loc[age, 'Estimat']:10.3f} "
          f"{age_results.loc[age, 'Std.feil']:10.3f} "
          f"{age_results.loc[age, 'P-verdi']:10.4f} "
          f"{age_results.loc[age, 'N']:8.0f}")