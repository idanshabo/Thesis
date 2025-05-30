import torch
import os
from Bio import SeqIO
from esm.models.esmc import ESMC
from esm.sdk.api import ESMCInferenceClient, ESMProtein, LogitsConfig, LogitsOutput

# Pad the sequence to a multiple of 16 (or 32, depending on model settings)
def pad_sequence(sequence_tensor, pad_length=16):
    current_length = sequence_tensor.size(1)
    if current_length % pad_length != 0:
        padding_size = pad_length - (current_length % pad_length)
        # Pad the sequence tensor with zeros or a padding token
        padding = torch.zeros((sequence_tensor.size(0), padding_size), device=sequence_tensor.device)
        sequence_tensor = torch.cat([sequence_tensor, padding], dim=1)
    return sequence_tensor

# Function to save embeddings with simple filenames
def sanitize_filename(name):
    # Remove or replace characters that are not safe in filenames
    return re.sub(r'[^A-Za-z0-9_\-\.]', '_', name)

def save_embeddings(embeddings, protein_name, save_dir):
    safe_name = sanitize_filename(protein_name)
    filename = os.path.join(save_dir, f"{safe_name}.pt")
    torch.save(embeddings, filename)
    print(f"Saved embeddings to {filename}")

def create_esm_embeddings_from_fasta(fasta_file, model, save_dir):
    sequences = []

    # Parse the FASTA file to get the sequences and names
    for record in SeqIO.parse(fasta_file, "fasta"):
        protein_name = record.id  # Or record.name or record.description depending on your needs
        sequences.append((protein_name, str(record.seq)))  # Store name with sequence

    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Process each sequence
    for protein_name, protein_sequence in sequences:
        print(f"Processing {protein_name}: {protein_sequence}")

        # Tokenize the sequence
        input_ids = model._tokenize([protein_sequence])[0]
        input_tensor = torch.tensor(input_ids).unsqueeze(0).long()

        # Pad
        padded_input = pad_sequence(input_tensor, pad_length=16)

        # Device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        padded_input = padded_input.to(device)

        # Inference
        with torch.no_grad():
            output = model(padded_input)

        logits, embeddings, hiddens = output.sequence_logits, output.embeddings, output.hidden_states

        # Save with name
        save_embeddings(embeddings, protein_name, save_dir)
        

if __name__ == "__main__":
    fasta_file = '/content/drive/MyDrive/protein_data/PF03618.fasta'
    save_dir = "embeddings_output"  
    model = ESMC.from_pretrained("esmc_300m")
    create_esm_embeddings_from_fasta(fasta_file, model, save_dir)
