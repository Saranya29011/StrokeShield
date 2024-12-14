from tkinter import ttk
import tkinter as tk
from tkinter import simpledialog
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import AgglomerativeClustering, KMeans
from PIL import Image, ImageTk

def perform_clustering():
    global data, attributes, data_scaled
    
    # Perform hierarchical clustering for all data
    distance_matrix = linkage(data_scaled, method='ward')
    
    # Perform hierarchical clustering for all data
    cluster_model = AgglomerativeClustering(n_clusters=3, linkage='ward')
    clusters = cluster_model.fit_predict(data_scaled)

    # Add cluster labels to the dataset
    data['cluster'] = clusters

    # Plot both dendrogram and hierarchical clustering in the same window
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    # Plot dendrogram
    dendrogram(distance_matrix, ax=ax1)
    ax1.set_title(' Hierarchial Dendrogram')
    ax1.set_xlabel('Individuals')
    ax1.set_ylabel('Distance')

    # Plot clusters based on age and glucose level
    for cluster_id in data['cluster'].unique():
        cluster_data = data[data['cluster'] == cluster_id]
        ax2.scatter(cluster_data['age'], cluster_data['avg_glucose_level'], label=f'Cluster {cluster_id}')

    # Add labels and legend to the hierarchical clustering plot
    ax2.set_title('Clustering of Individuals based on Age and Glucose Level')
    ax2.set_xlabel('Age')
    ax2.set_ylabel('Average Glucose Level')
    ax2.legend(title='Cluster')
    ax2.grid(True)

    # Plot histogram of age within clusters
    for cluster_id in data['cluster'].unique():
        cluster_data = data[data['cluster'] == cluster_id]
        ax3.hist(cluster_data['age'], bins=20, alpha=0.5, label=f'Cluster {cluster_id}', density=True)

    # Add labels and legend to the histogram plot
    ax3.set_title('Distribution of Age ')
    ax3.set_xlabel('Age')
    ax3.set_ylabel('Density')
    ax3.legend(title='Cluster')
    ax3.grid(True)

    # Show the plot
    plt.show()

def perform_kmeans_clustering():
    global data, attributes, data_scaled
    
    # Perform hierarchical clustering for all data
    distance_matrix = linkage(data_scaled, method='ward')
    
    # Perform hierarchical clustering for all data
    cluster_model = AgglomerativeClustering(n_clusters=3, linkage='ward')
    clusters = cluster_model.fit_predict(data_scaled)

    # Add cluster labels to the dataset
    data['cluster'] = clusters

    # Plot K-Means clustering results, hierarchical clustering, and histogram
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    # Plot K-Means clustering results
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans_clusters = kmeans.fit_predict(data_scaled)
    data['kmeans_cluster'] = kmeans_clusters
    ax1.scatter(data['age'], data['avg_glucose_level'], c=data['kmeans_cluster'], cmap='viridis')
    ax1.set_title('K-Means Clustering')
    ax1.set_xlabel('Age')
    ax1.set_ylabel('Average Glucose Level')
    ax1.grid(True)

    # Plot hierarchical clustering
    dendrogram(distance_matrix, ax=ax2)
    ax2.set_title('Hierarchical Clustering Dendrogram')
    ax2.set_xlabel('Individuals')
    ax2.set_ylabel('Distance')

    # Plot histogram of age within clusters
    for cluster_id in data['cluster'].unique():
        cluster_data = data[data['cluster'] == cluster_id]
        ax3.hist(cluster_data['age'], bins=20, alpha=0.5, label=f'Cluster {cluster_id}', density=True)

    # Add labels and legend to the histogram plot
    ax3.set_title('Distribution of Age within Clusters')
    ax3.set_xlabel('Age')
    ax3.set_ylabel('Density')
    ax3.legend(title='Cluster')
    ax3.grid(True)

    # Show the plot
    plt.show()


def select_range_for_clustering():
    global data, attributes, data_scaled

    
    
    # Prompt user to enter the number of rows for clustering
    num_rows = simpledialog.askinteger("Select Number of Rows", "Enter the number of rows for clustering:")
    
    # Handle invalid input or cancelation
    if num_rows is None or num_rows <= 0 or num_rows > len(data):
        return
    
    # Prompt user to select clustering method
    clustering_method = simpledialog.askstring("Select Clustering Method", "Enter 'hierarchical' or 'non-hierarchical' for clustering:")
    
    
    # Perform clustering based on user selection
    if clustering_method == 'hierarchical':
        # Select subset of data based on number of rows
        subset_data = data.head(num_rows).copy()  # Make a copy to avoid SettingWithCopyWarning
        subset_data_scaled = data_scaled[:num_rows]

        # Perform hierarchical clustering for the selected range
        distance_matrix = linkage(subset_data_scaled, method='ward')

        # Perform hierarchical clustering for the selected range
        cluster_model = AgglomerativeClustering(n_clusters=3, linkage='ward')
        clusters = cluster_model.fit_predict(subset_data_scaled)

        # Add cluster labels to the subset dataset
        subset_data.loc[:, 'cluster'] = clusters  # Use .loc to ensure modification on the original DataFrame

        # Plot both dendrogram and hierarchical clustering in the same window
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))

        # Plot dendrogram
        dendrogram(distance_matrix, ax=ax1)
        ax1.set_title('Hierarchical Clustering Dendrogram (Subset)')
        ax1.set_xlabel('Individuals')
        ax1.set_ylabel('Distance')

        # Plot clusters based on age and glucose level for the selected range
        for cluster_id in subset_data['cluster'].unique():
            cluster_data = subset_data[subset_data['cluster'] == cluster_id]
            ax2.scatter(cluster_data['age'], cluster_data['avg_glucose_level'], label=f'Cluster {cluster_id}')

        # Add labels and legend to the hierarchical clustering plot
        ax2.set_title('Hierarchical Clustering of Individuals based on Age and Glucose Level (Subset)')
        ax2.set_xlabel('Age')
        ax2.set_ylabel('Average Glucose Level')
        ax2.legend(title='Cluster')
        ax2.grid(True)

        # Plot histogram of age within clusters
        for cluster_id in subset_data['cluster'].unique():
            cluster_data = subset_data[subset_data['cluster'] == cluster_id]
            ax3.hist(cluster_data['age'], bins=20, alpha=0.5, label=f'Cluster {cluster_id}', density=True)

        # Add labels and legend to the histogram plot
        ax3.set_title('Distribution of Age within Clusters (Subset)')
        ax3.set_xlabel('Age')
        ax3.set_ylabel('Density')
        ax3.legend(title='Cluster')
        ax3.grid(True)

    elif clustering_method == 'non-hierarchical':
        # Perform non-hierarchical clustering (K-Means)
        kmeans = KMeans(n_clusters=3, random_state=42)
        kmeans_clusters = kmeans.fit_predict(data_scaled)

        # Add cluster labels to the dataset
        data['kmeans_cluster'] = kmeans_clusters

        # Plot K-Means clustering results
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))

        # Scatter plot for K-Means clustering
        scatter = ax1.scatter(data['age'], data['avg_glucose_level'], c=data['kmeans_cluster'], cmap='viridis')
        ax1.set_title('K-Means Clustering')
        ax1.set_xlabel('Age')
        ax1.set_ylabel('Average Glucose Level')
        fig.colorbar(scatter, ax=ax1, label='Cluster')
        ax1.grid(True)

        # Plot dendrogram for non-hierarchical clustering
        ax2.text(0.5, 0.5, 'Dendrogram not available\nfor K-Means', ha='center', va='center', fontsize=14, color='gray')
        ax2.axis('off')

        # Plot histogram of age within clusters for K-Means clustering
        for cluster_id in data['kmeans_cluster'].unique():
            cluster_data = data[data['kmeans_cluster'] == cluster_id]
            ax3.hist(cluster_data['age'], bins=20, alpha=0.5, label=f'Cluster {cluster_id}', density=True)
        ax3.set_title('Distribution of Age within Clusters (K-Means)')
        ax3.set_xlabel('Age')
        ax3.set_ylabel('Density')
        ax3.legend(title='Cluster')
        ax3.grid(True)

        # Show the plot
        plt.show()
        




# Load dataset and select relevant attributes
data = pd.read_csv("C:/Users/saran/Downloads/healthcare-dataset-stroke-data (1).csv")
attributes = ['age', 'avg_glucose_level']

# Handle missing values
imputer = SimpleImputer(strategy='mean')
data[attributes] = imputer.fit_transform(data[attributes])

# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[attributes])

# Create GUI
root = tk.Tk()
root.title("Clustering GUI")
root.state('zoomed')

root.focus_force()

# Set background image
background_image = Image.open("C:/2nd year/PA PROJECT/3.jpg")
background_photo = ImageTk.PhotoImage(background_image)
background_label = tk.Label(root, image=background_photo)
background_label.place(relwidth=1, relheight=1)

# Configure style for buttons
style = ttk.Style()
style.configure('Custom.TButton', font=('TIMES NEW ROMAN', 16),padding=(20, 10),background='lightblue',foreground='red')

# Create buttons with custom style
perform_button = ttk.Button(root, text="PERFORM CLUSTERING", command=perform_clustering, style='Custom.TButton')
perform_button.pack(pady=10)

select_button = ttk.Button(root, text="SELECT THE RANGE FOR CLUSTERING", command=select_range_for_clustering, style='Custom.TButton')
select_button.pack(pady=10)

kmeans_button = ttk.Button(root, text="Perform K-Means Clustering", command=perform_kmeans_clustering, style='Custom.TButton')
kmeans_button.pack(pady=10)

root.mainloop()
