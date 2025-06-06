def convert_embeddings_to_one_mean_embedding(folder_path, output_path=None):
    if not output_path:
        output_path = os.path.join(folder_path, 'mean_protein_embeddings.pt')
    # === Storage for mean embeddings ===
    mean_embeddings = []

    # === Process each .pt file one at a time ===
    for filename in os.listdir(folder_path):
        if filename.endswith('.pt'):
            filepath = os.path.join(folder_path, filename)
            
            with torch.no_grad():
                # Load the embedding tensor
                embedding = torch.load(filepath)  # assumes shape: (length, 960)
                
                # If it's a dict, get the actual tensor (adjust this if needed)
                if isinstance(embedding, dict):
                    embedding = embedding["representations"][33]  # example: layer 33
                
                # Compute mean embedding
                embedding = embedding.squeeze(0)
                mean_embedding = embedding.mean(dim=0)
                # Store the mean vector
                mean_embeddings.append(mean_embedding)
                file_names.append(filename)

                # Free memory
                del embedding
                del mean_embedding

    # === Stack all mean vectors into a final matrix ===
    all_mean_embeddings = torch.stack(mean_embeddings)  # shape: (num_proteins, 960)

    # === Save to file ===
    torch.save({
        'embeddings': all_mean_embeddings,
        'file_names': file_names
    }, output_path)

    # === Done ===
    print(f"✅ Processed {len(mean_embeddings)} proteins.")
    print(f"💾 Final tensor shape: {all_mean_embeddings.shape}")
    print(f"📁 Saved to: {output_path}")
    return(output_path)
