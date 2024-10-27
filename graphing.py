import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

pieces_str_lit = 'total_pieces'
price_str_lit = 'total_price'

def analyze_age_value(lego_sets) -> dict[int, dict[str, int]]:
    
    age_groups = {}
    
    for lego_set in lego_sets:
        age = lego_set.get_age_group()
        pieces = lego_set.get_pieces()
        price = lego_set.get_price()
        
        if age not in age_groups:
            age_groups[age] = {pieces_str_lit: 0, price_str_lit: 0}
            
        age_groups[age][pieces_str_lit] += pieces
        age_groups[age][price_str_lit] += price
    
    best_ratio = 0
    best_age = None
    
    for age, data in age_groups.items():
        if data[price_str_lit] > 0:  # 0 div exeption
            pieces_per_dollar = data[pieces_str_lit] / data[price_str_lit]
            if pieces_per_dollar > best_ratio:
                best_ratio = pieces_per_dollar
                best_age = age
    
    # debug
    for age, data in age_groups.items():
        if data[price_str_lit] > 0:
            ratio = data[pieces_str_lit] / data[price_str_lit]
            print(f"Age {age}+: {ratio:.2f} pieces per dollar")
    
    print(f"\nBest value: Age {best_age}+ with {best_ratio:.2f} pieces per dollar")
    
    return age_groups
    

def perform_cluster_analysis(lego_sets):
    data = []
    for lego_set in lego_sets:
        data.append([
            lego_set.get_age_group(),
            lego_set.get_price(),
            lego_set.get_pieces()
        ])
    
    X = np.array(data)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Elbow kurve
    inertias = []
    K = range(1, 10)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
    
    """
    plt.figure(figsize=(10, 6))
    plt.plot(K, inertias, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Inertia')
    plt.title('Elbow Method For Optimal k')
    plt.show()
    """
    
    # K means clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2],
                        c=clusters, cmap='viridis',
                        s=100)
    
    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    print(centers)
    
    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2],
              c='red', marker='x', s=200, linewidths=3,
              label='Cluster Centers')
    
    ax.set_xlabel('Age Group')
    ax.set_ylabel('Price')
    ax.set_zlabel('Pieces')
    ax.set_title('LEGO Sets Clusters')
    
    plt.colorbar(scatter)
    plt.legend()
    plt.show()
    
    df = pd.DataFrame(X, columns=['Age', 'Price', 'Pieces'])
    df['Cluster'] = clusters
    
    """
    print("\nCluster Analysis:")
    for i in range(kmeans.n_clusters):
        cluster_data = df[df['Cluster'] == i]
        print(f"\nCluster {i}:")
        print("Average Age:", cluster_data['Age'].mean())
        print("Average Price:", cluster_data['Price'].mean())
        print("Average Pieces:", cluster_data['Pieces'].mean())
        print("Number of sets:", len(cluster_data))
    """

    return clusters, df
    
    
def visualize_3d_regression(age_groups):
    ages = np.array([age for age in age_groups.keys()])
    prices = np.array([data[price_str_lit] for data in age_groups.values()])
    pieces = np.array([data[pieces_str_lit] for data in age_groups.values()])

    x_pred = np.linspace(min(ages), max(ages), 100)
    y_pred = np.linspace(min(prices), max(prices), 100)
    xx_pred, yy_pred = np.meshgrid(x_pred, y_pred)
    
    X = np.column_stack((ages, prices))
    y = pieces
    
    model = LinearRegression()
    model.fit(X, y)
    
    zz_pred = model.predict(np.c_[xx_pred.ravel(), yy_pred.ravel()]).reshape(xx_pred.shape)
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(ages, prices, pieces, c='red', marker='o', s=100, label='Data Points')
    
    surface = ax.plot_surface(xx_pred, yy_pred, zz_pred, alpha=0.3, cmap='viridis')
    
    ax.set_xlabel('Age Group')
    ax.set_ylabel('Total Price ($)')
    ax.set_zlabel('Total Pieces')
    ax.set_title('3D Regression Analysis of LEGO Sets')
    
    fig.colorbar(surface, ax=ax, shrink=0.5, aspect=5)
    
    
    # R-squared score
    r2_score = model.score(X, y)
    plt.figtext(0.02, 0.02, f'R² Score: {r2_score:.3f}', fontsize=10)
    
    equation = f'Pieces = {model.coef_[0]:.2f}*Age + {model.coef_[1]:.2f}*Price + {model.intercept_:.2f}'
    plt.figtext(0.02, 0.05, equation, fontsize=10)
    
    plt.show()