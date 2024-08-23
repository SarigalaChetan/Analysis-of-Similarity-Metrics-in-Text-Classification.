import io
import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import jensenshannon
import tkinter as tk
from tkinter import filedialog
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
import matplotlib.pyplot as plt
from kneed import KneeLocator

def create_dataframe(matrix, doc_names, tokens):
    df = pd.DataFrame(data=matrix, index=doc_names, columns=tokens)
    return df

def simrank_similarity(matrix, num_iterations=10, decay_factor=0.8):
    num_docs = matrix.shape[0]
    sim_matrix = np.identity(num_docs)
    for i in range(num_iterations):
        new_sim_matrix = np.zeros_like(sim_matrix)
        for j in range(num_docs):
            for k in range(num_docs):
                if j == k:
                    new_sim_matrix[j, k] = 1.0
                else:
                    s1 = [l for l in range(num_docs) if matrix[j, l]]
                    s2 = [l for l in range(num_docs) if matrix[k, l]]
                    numerator = decay_factor * sum(sim_matrix[p][q] for p in s1 for q in s2) if s1 and s2 else 0.0
                    denominator = len(s1) * len(s2)
                    new_sim_matrix[j, k] = numerator / denominator if denominator != 0 else 0.0
        sim_matrix = new_sim_matrix
    return sim_matrix




root = tk.Tk()
root.withdraw()
dir_path = filedialog.askdirectory()

docs = []
doc_names = []
for i, file_name in enumerate(os.listdir(dir_path)):
    doc_name = f"doc_{i+1}"

    file_path = os.path.join(dir_path, file_name)
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        doc = file.read()

    docs.append(doc)
    doc_names.append(doc_name)

# Create a CountVectorizer object and transform the data
count_vectorizer = CountVectorizer()
vector_matrix = count_vectorizer.fit_transform(docs)

# Get the feature names and create a dataframe
tokens = count_vectorizer.get_feature_names()
df_vector_matrix = create_dataframe(vector_matrix.toarray(), doc_names, tokens)

# Compute cosine similarity between the documents
cosine_similarity_matrix = cosine_similarity(vector_matrix)
df_cos_sim = create_dataframe(cosine_similarity_matrix, doc_names, doc_names)

# Compute Jaccard similarity between the documents
jaccard_similarity_matrix = pairwise_distances(vector_matrix.toarray(), metric='jaccard')
df_jaccard_sim = create_dataframe(1 - jaccard_similarity_matrix, doc_names, doc_names)

# Compute Euclidean distance between the documents
euclidean_distance = euclidean_distances(vector_matrix)
df_euclidean_dist = create_dataframe(euclidean_distance, doc_names, doc_names)

# Compute SimRank similarity between the documents
simrank_similarity_matrix = simrank_similarity(vector_matrix.toarray())
df_simrank_sim = create_dataframe(simrank_similarity_matrix, doc_names, doc_names)

# Compute Jenson-Shannon similarity between the documents
jensenshannon_similarity_matrix = pairwise_distances(vector_matrix.toarray(), metric=jensenshannon)
df_jensenshannon_sim = create_dataframe(1 - jensenshannon_similarity_matrix, doc_names, doc_names)

# Compute Dice similarity between the documents
dice_similarity_matrix = pairwise_distances(vector_matrix.toarray(), metric='dice')
df_dice_sim = create_dataframe(1 - dice_similarity_matrix, doc_names, doc_names)

# Compute Manhattan similarity between the documents
manhattan_distance = manhattan_distances(vector_matrix)
df_manhattan_sim = create_dataframe(1 / (1 + manhattan_distance), doc_names, doc_names)


# Calculate the within-cluster sum of squares (WCSS) for different values of k
wcss = []
max_clusters = 10 
for k in range(1, max_clusters + 1):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(vector_matrix)
    wcss.append(kmeans.inertia_)

# Plot the WCSS values against the number of clusters
plt.plot(range(1, max_clusters + 1), wcss, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('Elbow Method')
plt.show()

# Determine the optimal number of clusters using the elbow method
knee = KneeLocator(range(1, max_clusters + 1), wcss, curve='convex', direction='decreasing')
optimal_clusters = knee.elbow
print(optimal_clusters)

# Perform K-means clustering with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
kmeans.fit(vector_matrix)

# Get the cluster labels
cluster_labels = kmeans.labels_
print(cluster_labels)
# Calculate the Davies-Bouldin index for each similarity method
davies_bouldin_scores = []

similarity_matrices = [
    cosine_similarity_matrix,
    1 - jaccard_similarity_matrix,
    euclidean_distance,
    simrank_similarity_matrix,
    1 - jensenshannon_similarity_matrix,
    1 - dice_similarity_matrix,
    1 / (1 + manhattan_distance)
]

similarity_methods = [
    "Cosine Similarity",
    "Jaccard Similarity",
    "Euclidean Distance",
    "SimRank Similarity",
    "Jenson-Shannon Similarity",
    "Dice Similarity",
    "Manhattan Similarity"
]

for similarity_matrix, similarity_method in zip(similarity_matrices, similarity_methods):
    kmeans.fit(similarity_matrix)
    cluster_labels = kmeans.labels_
    db_score = davies_bouldin_score(similarity_matrix, cluster_labels)
    davies_bouldin_scores.append(db_score)

print("\nTerm frequency count matrix:")
print(create_dataframe(vector_matrix.toarray(), doc_names, tokens))
print("\nCosine similarity matrix:")
print(df_cos_sim)
print("\nJaccard similarity matrix:")
print(df_jaccard_sim)
print("\nEuclidean distance matrix:")
print(df_euclidean_dist)
print("\nSimRank similarity matrix:")
print(df_simrank_sim)
print("\nJenson-Shannon Similarity matrix:")
print(df_jensenshannon_sim)
print("\nDice similarity matrix:")
print(df_dice_sim)
print("\nManhattan similarity matrix:")
print(df_manhattan_sim)

print("\nDavies-Bouldin Index:")
for similarity_method, db_score in zip(similarity_methods, davies_bouldin_scores):
    print(f"{similarity_method}: {db_score}")

