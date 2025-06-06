import torch
from scgpt.model import TransformerModel
import json
import os
import pandas as pd
import numpy as np

def load_model(model_path, args_path, vocab_path):
    # Load model configuration
    with open(args_path, 'r') as f:
        model_args = json.load(f)
    
    # Load vocabulary
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    
    # Initialize model
    model = TransformerModel(
        vocab_size=len(vocab),
        **model_args
    )
    
    # Load the trained weights
    state_dict = torch.load(model_path, map_location='cpu')
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    model.load_state_dict(state_dict)
    model.eval()
    
    return model, vocab

def get_embeddings(model, genes, vocab):
    """Extract embeddings for a list of genes."""
    model.eval()
    with torch.no_grad():
        # Convert genes to indices using vocabulary
        gene_indices = torch.tensor([vocab.get(gene, vocab['<unk>']) for gene in genes])
        # Get embeddings
        embeddings = model.encoder.embedding(gene_indices)
    return embeddings

def save_embeddings(embeddings, genes, output_file):
    """Save embeddings to a file along with their corresponding gene names."""
    # Convert embeddings to numpy array
    embeddings_np = embeddings.numpy()
    
    # Create a dictionary with gene names as keys and embeddings as values
    embeddings_dict = {
        'gene': genes,
        **{f'dim_{i}': embeddings_np[:, i] for i in range(embeddings_np.shape[1])}
    }
    
    # Convert to DataFrame and save
    df = pd.DataFrame(embeddings_dict)
    df.to_csv(output_file, index=False)
    print(f"Embeddings saved to {output_file}")
    
    # Also save as JSON for compatibility
    embeddings_json = {gene: emb.tolist() for gene, emb in zip(genes, embeddings_np)}
    with open(output_file.replace('.csv', '.json'), 'w') as f:
        json.dump(embeddings_json, f)
    print(f"Embeddings also saved to {output_file.replace('.csv', '.json')}")

if __name__ == "__main__":
    # Paths to model files
    MODEL_PATH = "model_files/best_model.pt"
    ARGS_PATH = "model_files/args.json"
    VOCAB_PATH = "model_files/vocab.json"
    
    # Your specific genes
    genes = [
        "USP44", "PRKACG", "NDUFB3", "NDUFV2", "NDUFB6", "MRPL13", "ESR2", 
        "TFAP2A", "MAPT", "USP49", "TNNI3", "NDUFS4", "ALPP", "ALPI", 
        "GNA14", "KRT10", "SLC4A4", "ZNF430", "CA4", "ZNF223", "PLK3", 
        "GSTM4", "TNNT1", "TDRD1", "FFAR2"
    ]
    
    # Load the model
    print("Loading model...")
    model, vocab = load_model(MODEL_PATH, ARGS_PATH, VOCAB_PATH)
    print("Model loaded successfully!")
    
    # Check which genes are in vocabulary
    genes_in_vocab = [gene for gene in genes if gene in vocab]
    genes_not_in_vocab = [gene for gene in genes if gene not in vocab]
    
    if genes_not_in_vocab:
        print("\nWarning: The following genes are not in the vocabulary:")
        print(", ".join(genes_not_in_vocab))
        print("\nThey will be represented using the <unk> token.")
    
    # Get embeddings
    print("\nExtracting embeddings...")
    embeddings = get_embeddings(model, genes, vocab)
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Save embeddings in both CSV and JSON formats
    output_file = "gene_embeddings.csv"
    save_embeddings(embeddings, genes, output_file) 