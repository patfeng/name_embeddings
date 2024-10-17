from openai import OpenAI
import os
import json
from tqdm import tqdm
import h5py
import numpy as np

client = OpenAI()


def create_name_frequency_prompt(filename):
    prompt = []
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                rank, name, frequency = parts[0], parts[1], parts[2]
                frequency = int(frequency.replace(',', ''))
                prompt.append([name,frequency])
    return prompt

# Specify the input file name
input_file = 'names.txt'

try:
    prompt = create_name_frequency_prompt(input_file)
except FileNotFoundError:
    print(f"Error: The file '{input_file}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")

# Add OpenAI API key
client.api_key = os.getenv("OPENAI_API_KEY")

# Function to get embeddings in batches
def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input = text, model=model).data[0].embedding
    
def get_embeddings_batch(texts, batch_size=100, model="text-embedding-3-small"):
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        embeddings = [get_embedding(text, model) for text in batch]
        yield embeddings

# Prepare the names for batch processing
names = [p[0] for p in prompt]
frequencies = [p[1] for p in prompt]

# Open the HDF5 file for writing
with h5py.File('name_embeddings.h5', 'w') as h5_file:
    pass  # Just create the file, we'll add datasets later

# Process embeddings in batches and write to file
for i, batch_embeddings in enumerate(get_embeddings_batch(names)):
    start_idx = i * 100  # Assuming batch_size=100
    end_idx = start_idx + len(batch_embeddings)
    
    with h5py.File('name_embeddings.h5', 'a') as h5_file:
        for j, embedding in enumerate(batch_embeddings):
            idx = start_idx + j
            name = names[idx]
            frequency = frequencies[idx]
            
            # Create a dataset for each name
            # The dataset will contain the embedding and the frequency
            data = np.concatenate([embedding, [frequency]])
            h5_file.create_dataset(name, data=data)

print("Embeddings and frequencies saved to name_embeddings.h5")
