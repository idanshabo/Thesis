import torch
import os
import re
from Bio import SeqIO
from esm.models.esmc import ESMC
from transformers import T5Tokenizer, T5EncoderModel

def sanitize_filename(name):
    return re.sub(r'[^A-Za-z0-9_\-\.]', '_', name)

def pad_sequence(sequence_tensor, pad_length=16):
    current_length = sequence_tensor.size(1)
    if current_length % pad_length != 0:
        padding_size = pad_length - (current_length % pad_length)
        padding = torch.full(
            (sequence_tensor.size(0), padding_size),
            fill_value=0, 
            device=sequence_tensor.device,
            dtype=sequence_tensor.dtype
        )
        sequence_tensor = torch.cat([sequence_tensor, padding], dim=1)
    return sequence_tensor

def generate_embeddings(fasta_file, output_path, mode="sequence"):
    """
    mode: str, either "sequence" (ESMC) or "structure" (ProstT5)
    """
    os.makedirs(output_path, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sequences = [(rec.id, str(rec.seq)) for rec in SeqIO.parse(fasta_file, "fasta")]

    if mode == "sequence":
        print("Loading ESMC model for sequence-directed embeddings...")
        model = ESMC.from_pretrained("esmc_300m").to(device)
        model.eval()
        
        for protein_name, protein_sequence in sequences:
            output_file = os.path.join(output_path, f"{sanitize_filename(protein_name)}.pt")
            if os.path.exists(output_file): continue
                
            input_ids = model._tokenize([protein_sequence])[0]
            input_tensor = input_ids.clone().detach().unsqueeze(0).long()
            padded_input = pad_sequence(input_tensor, pad_length=16).to(device)
            
            with torch.no_grad():
                output = model(padded_input)
            
            # Save the sequence embeddings
            torch.save(output.embeddings.cpu(), output_file)
            print(f"Saved ESMC embedding for {protein_name}")

    elif mode == "structure":
        print("Loading ProstT5 model for structure-directed embeddings...")
        tokenizer = T5Tokenizer.from_pretrained('Rostlab/ProstT5', do_lower_case=False)
        model = T5EncoderModel.from_pretrained("Rostlab/ProstT5").to(device)
        model.eval()

        for protein_name, protein_sequence in sequences:
            output_file = os.path.join(output_path, f"{sanitize_filename(protein_name)}.pt")
            if os.path.exists(output_file): continue
            
            # ProstT5 requires spaces between amino acids
            # <AA2fold> forces the model to generate structure-directed representations
            seq_spaced = " ".join(list(protein_sequence.upper()))
            full_prompt = f"<AA2fold> {seq_spaced}"
            
            inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                
            # ProstT5 outputs shape: (batch, seq_len, hidden_dim)
            # We slice [0, 1:-1, :] to remove the <AA2fold> prompt token and the </s> EOS token
            embeddings = outputs.last_hidden_state[0, 1:-1, :]
            
            torch.save(embeddings.cpu(), output_file)
            print(f"Saved ProstT5 embedding for {protein_name}")
            
    else:
        raise ValueError("Mode must be 'sequence' or 'structure'")
