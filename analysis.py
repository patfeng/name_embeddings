import h5py
import numpy as np
from scipy.special import softmax
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def l2_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def kl_divergence(p, q):
    p = softmax(p)
    q = softmax(q)
    return np.sum(p * np.log(p / q))

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Open the HDF5 file
with h5py.File('name_embeddings.h5', 'r') as f:
    # Get the list of names (keys) in the file
    names = list(f.keys())
    
    # Print the total number of names in the file
    print(f"Total names in the file: {len(names)}")

    # Get the shape of the first embedding to initialize the total_weighted_embedding
    first_name = names[0]
    first_data = f[first_name][:]
    embedding_shape = first_data[:-1].shape

    print(f"Shape of the first embedding: {embedding_shape}")

    # Calculate weighted average embedding
    total_weighted_embedding = np.zeros(embedding_shape)
    total_frequency = 0

    for name in names:
        data = f[name][:]
        embedding = data[:-1]
        frequency = int(data[-1])
        
        if embedding.shape != embedding_shape:
            print(f"Warning: Embedding shape mismatch for name '{name}'. Expected {embedding_shape}, got {embedding.shape}")
            continue

        total_weighted_embedding += embedding * frequency
        total_frequency += frequency

    average_embedding = total_weighted_embedding / total_frequency

    # Find the name with the closest and farthest embedding (L2 distance)
    closest_name_l2, farthest_name_l2 = None, None
    min_distance_l2, max_distance_l2 = float('inf'), float('-inf')

    # Find the name with the closest and farthest embedding (KL divergence)
    closest_name_kl, farthest_name_kl = None, None
    min_distance_kl, max_distance_kl = float('inf'), float('-inf')

    # Find the name with the closest and farthest embedding (cosine similarity)
    closest_name_cos, farthest_name_cos = None, None
    max_similarity_cos, min_similarity_cos = float('-inf'), float('inf')

    for name in names:
        data = f[name][:]
        embedding = data[:-1]
        
        if embedding.shape != embedding_shape:
            continue

        distance_l2 = l2_distance(embedding, average_embedding)
        distance_kl = kl_divergence(embedding, average_embedding)
        similarity_cos = cosine_similarity(embedding, average_embedding)
        
        if distance_l2 < min_distance_l2:
            min_distance_l2 = distance_l2
            closest_name_l2 = name
        if distance_l2 > max_distance_l2:
            max_distance_l2 = distance_l2
            farthest_name_l2 = name

        if distance_kl < min_distance_kl:
            min_distance_kl = distance_kl
            closest_name_kl = name
        if distance_kl > max_distance_kl:
            max_distance_kl = distance_kl
            farthest_name_kl = name

        if similarity_cos > max_similarity_cos:
            max_similarity_cos = similarity_cos
            closest_name_cos = name
        if similarity_cos < min_similarity_cos:
            min_similarity_cos = similarity_cos
            farthest_name_cos = name

    # Print results
    print("\nResults:")
    print(f"Weighted average embedding shape: {average_embedding.shape}")
    print(f"First 5 values of weighted average embedding: {average_embedding[:5]}")
    
    print(f"\nName with closest embedding (L2 distance): {closest_name_l2}")
    print(f"L2 distance to average: {min_distance_l2}")
    print(f"Name with farthest embedding (L2 distance): {farthest_name_l2}")
    print(f"L2 distance to average: {max_distance_l2}")

    print(f"\nName with closest embedding (KL divergence): {closest_name_kl}")
    print(f"KL divergence to average: {min_distance_kl}")
    print(f"Name with farthest embedding (KL divergence): {farthest_name_kl}")
    print(f"KL divergence to average: {max_distance_kl}")

    print(f"\nName with closest embedding (cosine similarity): {closest_name_cos}")
    print(f"Cosine similarity to average: {max_similarity_cos}")
    print(f"Name with farthest embedding (cosine similarity): {farthest_name_cos}")
    print(f"Cosine similarity to average: {min_similarity_cos}")

    # Print details of the closest name (L2 distance)
    closest_data_l2 = f[closest_name_l2][:]
    closest_embedding_l2 = closest_data_l2[:-1]
    closest_frequency_l2 = int(closest_data_l2[-1])

    print(f"\nClosest name details (L2 distance):")
    print(f"Name: {closest_name_l2}")
    print(f"Frequency: {closest_frequency_l2}")

    # Print details of the farthest name (L2 distance)
    farthest_data_l2 = f[farthest_name_l2][:]
    farthest_embedding_l2 = farthest_data_l2[:-1]
    farthest_frequency_l2 = int(farthest_data_l2[-1])

    print(f"\nFarthest name details (L2 distance):")
    print(f"Name: {farthest_name_l2}")
    print(f"Frequency: {farthest_frequency_l2}")

    # Print details of the closest name (KL divergence)
    closest_data_kl = f[closest_name_kl][:]
    closest_embedding_kl = closest_data_kl[:-1]
    closest_frequency_kl = int(closest_data_kl[-1])

    print(f"\nClosest name details (KL divergence):")
    print(f"Name: {closest_name_kl}")
    print(f"Frequency: {closest_frequency_kl}")

    # Print details of the farthest name (KL divergence)
    farthest_data_kl = f[farthest_name_kl][:]
    farthest_embedding_kl = farthest_data_kl[:-1]
    farthest_frequency_kl = int(farthest_data_kl[-1])

    print(f"\nFarthest name details (KL divergence):")
    print(f"Name: {farthest_name_kl}")
    print(f"Frequency: {farthest_frequency_kl}")

    # Print details of the closest name (cosine similarity)
    closest_data_cos = f[closest_name_cos][:]
    closest_embedding_cos = closest_data_cos[:-1]
    closest_frequency_cos = int(closest_data_cos[-1])

    print(f"\nClosest name details (cosine similarity):")
    print(f"Name: {closest_name_cos}")
    print(f"Frequency: {closest_frequency_cos}")

    # Print details of the farthest name (cosine similarity)
    farthest_data_cos = f[farthest_name_cos][:]
    farthest_embedding_cos = farthest_data_cos[:-1]
    farthest_frequency_cos = int(farthest_data_cos[-1])

    print(f"\nFarthest name details (cosine similarity):")
    print(f"Name: {farthest_name_cos}")
    print(f"Frequency: {farthest_frequency_cos}")

    # Perform t-SNE
    print("\nPerforming t-SNE...")
    embeddings = []
    names_for_tsne = []
    frequencies = []

    for name in names:
        data = f[name][:]
        embedding = data[:-1]
        frequency = int(data[-1])
        
        if embedding.shape != embedding_shape:
            continue

        embeddings.append(embedding)
        names_for_tsne.append(name)
        frequencies.append(frequency)

    embeddings_array = np.array(embeddings)
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings_array)

    # Sort names by frequency and get top 200
    top_200_indices = np.argsort(frequencies)[-200:]
    top_200_names = [names_for_tsne[i] for i in top_200_indices]

    # Visualize t-SNE results
    plt.figure(figsize=(24, 20))  # Increased figure size even more
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                          c=np.log(frequencies), cmap='viridis', 
                          alpha=0.6, s=50)
    plt.colorbar(scatter, label='Log Frequency')
    plt.title('t-SNE visualization of name embeddings (Top 200 labeled)', fontsize=20)
    plt.xlabel('t-SNE dimension 1', fontsize=16)
    plt.ylabel('t-SNE dimension 2', fontsize=16)

    # Annotate top 200 names with larger font size
    for idx in top_200_indices:
        plt.annotate(names_for_tsne[idx], (embeddings_2d[idx, 0], embeddings_2d[idx, 1]), 
                     fontsize=12, alpha=0.8)

    plt.tight_layout()
    plt.savefig('tsne_visualization_top200.png', dpi=300, bbox_inches='tight')
    print("t-SNE visualization saved as 'tsne_visualization_top200.png'")

# Close the h5py file (it's good practice to explicitly close it)
f.close()
