import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression

# Sett norske fonter
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

# Last og preparer data
df = pd.read_csv('DATA/lego.population.csv', encoding='latin1')

# Konverter priskolonner til numeriske verdier
df['Price'] = pd.to_numeric(df['Price'].str.replace('$', '').str.replace(',', ''), errors='coerce')
df['Amazon_Price'] = pd.to_numeric(df['Amazon_Price'].str.replace('$', '').str.replace(',', ''), errors='coerce')
df['Final_Price'] = df['Price'].fillna(df['Amazon_Price'])
df['Pieces'] = pd.to_numeric(df['Pieces'], errors='coerce')

# Fjern missing verdier
mask = df['Pieces'].notna() & df['Final_Price'].notna()
X = df.loc[mask, 'Pieces'].values.reshape(-1, 1)
y = df.loc[mask, 'Final_Price'].values

# Tilpass modell
model = LinearRegression()
model.fit(X, y)

# Beregn prediksjoner og residualer
y_pred = model.predict(X)
residuals = y - y_pred

# Opprett subplot figur
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# QQ-plot
stats.probplot(residuals, dist="norm", plot=ax1)
ax1.set_title("Q-Q Plot av residualer")
ax1.set_xlabel("Teoretiske kvantiler")
ax1.set_ylabel("Standardiserte residualer")

# Fordelingsplot av residualer
sns.histplot(residuals, kde=True, ax=ax2)
ax2.axvline(x=0, color='r', linestyle='--', alpha=0.5)
ax2.set_xlabel("Residualer")
ax2.set_ylabel("Antall observasjoner")
ax2.set_title("Fordeling av residualer")

# Juster layo
plt.tight_layout()

# Vis plot
plt.show()

# Lagre plottet med høy oppløsning
plt.savefig('qq_residual_distribution.png', dpi=300, bbox_inches='tight')