

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt  
import seaborn as sns
import warnings


def assign_cluster_dbscan(X,epsilon, customer_data):
    
    from sklearn.cluster import DBSCAN
    
    clusters = DBSCAN(eps = epsilon).fit_predict(X)
    customer_data['Cluster'] =  clusters
    
    return customer_data
    
    


def assign_cluster(X, num_clusters, customer_data):
    
    from sklearn.cluster import KMeans

    main_model = KMeans(n_clusters=num_clusters, random_state=42)

    Y = main_model.fit_predict(X)
    
    customer_data['Cluster'] = Y

    return customer_data


def visualize_numerical_features(customer_data):


    customer_data_grouped = customer_data.select_dtypes(include = 'number')

    # Get the number of columns in the DataFrame
    num_columns = len(customer_data_grouped.columns)

    # Calculate the number of rows needed to display 2 graphs on each row
    num_rows = (num_columns + 1) // 2

    # Create subplots with the specified number of rows and 2 columns per row
    fig, axes = plt.subplots(nrows=num_rows, ncols=2, figsize=(20, 20))

    # Flatten the axes array for easier indexing
    axes = axes.flatten()

    # Plot each column as a bar plot using sns.barplot
    for i, column in enumerate(customer_data_grouped.columns):
        ax = axes[i]
        sns.barplot(data=customer_data_grouped, x=customer_data_grouped.index, y=column, ax=ax)
        ax.set_title(column)
        ax.set_xlabel('Cluster')  # Label x-axis as 'Cluster'
        ax.set_ylabel(column)     # Label y-axis with the respective column name

    # Hide any empty subplots
    for i in range(num_columns, len(axes)):
        fig.delaxes(axes[i])

    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()
    
    return None


def visualize_categorical_features(customer_data):
    
    cat_columns = customer_data.select_dtypes(include=['object'])

    fig = plt.figure(figsize=(18, 6))
    for i, col in enumerate(cat_columns):
        plot_df = pd.crosstab(index=customer_data['Cluster'], columns=customer_data[col], values=customer_data[col], aggfunc='size', normalize='index')
        ax = fig.add_subplot(1, 3, i+1)
        plot_df.plot.bar(stacked=True, ax=ax, alpha=0.6)
        ax.set_title(f'% {col.title()} per Cluster', alpha=0.5)

        ax.set_ylim(0, 1.4)
        ax.legend(frameon=False)
        ax.xaxis.grid(False)
        
        labels = [0, 0.2, 0.4, 0.6, 0.8, 1]
        ax.set_yticklabels(labels)

    plt.tight_layout()
    
    plt.show()
    
    return None
        
    