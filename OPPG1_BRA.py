import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# Les inn datasettet
df = pd.read_csv("Data/lego.population.csv", sep=",", encoding="latin1")

# Funksjon for å standardisere aldersgrupper
def standardize_age_old(age_str):
    if pd.isna(age_str):
        return 'Unknown'
    
    age_str = str(age_str).lower().replace('ages_', '').replace('age_', '')
    
    try:
        if '+' in age_str:
            base_age = age_str.replace('+', '')
            return f"{int(float(base_age))}+"
        elif '-' in age_str:
            start, end = map(lambda x: int(float(x)), age_str.split('-'))
            return f"{start}-{end}"
        else:
            return str(int(float(age_str)))
    except:
        return 'Unknown'
        
def standardize_age(age_str):
    if pd.isna(age_str):
        return '-1'
    
    age_str = str(age_str).lower().replace('ages_', '').replace('age_', '')
    
    try:
        if '+' in age_str:
            base_age = int(float(age_str.replace('+', '')))
            return f"{base_age}+"
        elif '-' in age_str:
            start, end = map(lambda x: int(float(x)), age_str.split('-'))
            return f"{start}+"
        else:
            age = int(float(age_str))
            return str(age)+""
    except:
        return '-1'

# Anvendt standardisering
df['Age_Group'] = df['Ages'].apply(standardize_age)

# Beregn verdi-for-pengene metrics
df['Price_per_Piece'] = df['Price'].str.replace('$', '').astype(float) / df['Pieces']
df['Price_per_Unique'] = df['Price'].str.replace('$', '').astype(float) / df['Unique_Pieces']

# 3D Scatter plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Fargekoding for aldersgrupper
age_groups = sorted(df['Age_Group'].unique(), key=lambda x: int(x.replace('+', '')) if x != '-1' else -1)
colors = plt.cm.rainbow(np.linspace(0, 1, len(age_groups)))

for age, color in zip(age_groups, colors):
    mask = df['Age_Group'] == age
    ax.scatter(df[mask]['Pieces'], 
              df[mask]['Price'].str.replace('$', '').astype(float),
              df[mask]['Unique_Pieces'],
              c=[color], label=age)

ax.set_xlabel('Totalt antall brikker')
ax.set_ylabel('Pris ($)')
ax.set_zlabel('Unike brikker')
plt.title('Lego sett: Pris vs. Brikker vs. Unike brikker per aldersgruppe')
plt.legend(bbox_to_anchor=(1.15, 1))
plt.tight_layout()
plt.show()

# Box plot for pris per brikke per aldersgruppe
plt.figure(figsize=(12, 6))
sns.boxplot(x='Age_Group', y='Price_per_Piece', data=df, 
            order=sorted(df['Age_Group'].unique(), key=lambda x: int(x.replace('+', '')) if x != '-1' else -1))
plt.xticks(rotation=45)
plt.title('Pris per brikke for hver aldersgruppe')
plt.tight_layout()
plt.show()

# Heatmap for gjennomsnittsverdier
pivot_table = df.pivot_table(
    values=['Price_per_Piece', 'Price_per_Unique'],
    index='Age_Group',
    aggfunc='mean'
).reindex(sorted(df['Age_Group'].unique(), key=lambda x: int(x.replace('+', '')) if x != '-1' else -1))

plt.figure(figsize=(10, 8))
sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='YlOrRd')
plt.title('Gjennomsnittlig pris per brikke/unik brikke per aldersgruppe')
plt.tight_layout()
plt.show()