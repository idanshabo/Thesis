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
def save_embeddings(embeddings, index, save_dir):
    # Save embeddings with a simple filename like "prot1.pt", "prot2.pt", etc.
    filename = os.path.join(save_dir, f"prot{index}.pt")
    torch.save(embeddings, filename)
    print(f"Saved embeddings to {filename}")

def calculate_embeddings_for_all_proteins_in_msa(fasta_file, save_dir, model=ESMC.from_pretrained("esmc_300m")):
    sequences = []
    # Parse the FASTA file to get the sequences
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequences.append(str(record.seq))  # Append the sequence as a string

    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Process each sequence
    for index, protein_sequence in enumerate(sequences, 1):  # Start numbering from 1
        print(f"Processing sequence: {protein_sequence}")
        # Tokenize the sequence using the model's tokenizer
        input_ids = model._tokenize([protein_sequence])[0]  # Tokenize a list of sequences

        # Convert to tensor and ensure proper padding
        input_tensor = torch.tensor(input_ids).unsqueeze(0).long()  # Add batch dimension and convert to Long type
        print(f"Input tensor shape before padding: {input_tensor.shape}")

        # Pad the sequence to a multiple of 16 (or 32)
        padded_input = pad_sequence(input_tensor, pad_length=16)  # Pad to 16-length
        print(f"Input tensor shape after padding: {padded_input.shape}")

        # Ensure device compatibility (move model and tensor to GPU if available)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        padded_input = padded_input.to(device)  # Move tensor to the same device as the model

        # Double-check tensor type (make sure it's LongTensor)
        if padded_input.dtype != torch.long:
            padded_input = padded_input.long()  # Ensure the tensor is Long type

        # Forward pass through the model
        with torch.no_grad():
            output = model(padded_input)

        # Extract logits, embeddings, and hidden states
        logits, embeddings, hiddens = (
            output.sequence_logits,
            output.embeddings,
            output.hidden_states,
        )

        # Print the shapes of the results
        print(
            f"Logits shape: {logits.shape}, "
            f"Embeddings shape: {embeddings.shape}, "
            f"Hidden states shape: {hiddens.shape}"
        )

        # Save embeddings with a simple filename
        save_embeddings(embeddings, index, save_dir)
