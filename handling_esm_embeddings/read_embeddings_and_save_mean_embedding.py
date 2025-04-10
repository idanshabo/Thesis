import torch
import os

def load_embedding(file_path):
    if torch.cuda.is_available():
        embedding = torch.load(file)
    else:
        embedding = torch.load(file, map_location=torch.device('cpu'))
    return embedding

def calculate_mean_embedding_per_protein(all_embeddings):
    # Convert the list of embeddings to a tensor
    all_embeddings_tensor = torch.stack(all_embeddings)

    # Squeeze the tensor if needed to remove the extra dimension
    #all_embeddings_tensor = all_embeddings_tensor.squeeze(1)

    # Compute the mean embedding along the batch dimension (dim=0)
    mean_embedding = torch.mean(all_embeddings_tensor, dim=0)

    return mean_embedding

def save_embeddings(embeddings, output_path):
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, 'mean_embedding.pt')
    torch.save(embedding, output_file)
    print(f"Mean embedding saved to: {output_file}")


def read_embeddings_and_save_mean_embedding(folder_path, output_path):
    all_embeddings = []

    # Iterate over all .pt files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.pt'):
            file = os.path.join(folder_path, file_name)
            
            embedding = load_embedding(file)
            all_embeddings.append(embedding)

    mean_embedding = calculate_mean_embedding_per_protein(all_embeddings)

    save_embeddings(mean_embedding, output_path)
    return mean_embedding
