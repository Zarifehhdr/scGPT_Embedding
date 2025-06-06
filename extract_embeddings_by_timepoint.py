import torch
from scgpt.model import TransformerModel
import json
import os
import pandas as pd
import numpy as np

def load_expression_data(file_path):
    """Load gene expression data from file."""
    # Read the tab-separated file
    df = pd.read_csv(file_path, sep='\t', index_col='time')
    
    # Convert to dictionary format
    expression_data = {}
    for timepoint in df.index:
        expression_data[timepoint] = {
            "genes": list(df.columns),
            "expressions": df.loc[timepoint].values.tolist()
        }
    
    return expression_data

def load_model(model_path, args_path, vocab_path):
    # Load model configuration
    with open(args_path, 'r') as f:
        model_args = json.load(f)
    
    # Load vocabulary
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    
    # Initialize model with the correct arguments from args.json
    model = TransformerModel(
        ntoken=len(vocab),
        d_model=model_args.get('embsize', 512),
        nhead=model_args.get('nheads', 8),
        d_hid=model_args.get('d_hid', 512),
        nlayers=model_args.get('nlayers', 12),
        dropout=model_args.get('dropout', 0.2),
        vocab=vocab,
        use_fast_transformer=model_args.get('fast_transformer', True),
        pad_token=model_args.get('pad_token', '<pad>'),
        do_mvc=model_args.get('MVC', True),
        input_emb_style="continuous"  # Use continuous embedding for expression values
    )
    
    # Load the trained weights
    state_dict = torch.load(model_path, map_location='cpu')
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    
    # Remove unexpected keys from state_dict
    model_state_dict = model.state_dict()
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict}
    
    # Load filtered state dict
    model.load_state_dict(filtered_state_dict, strict=False)
    model.eval()
    
    return model, vocab

def get_embeddings(model, genes, expression_values, vocab):
    """Extract embeddings for a list of genes with their expression values."""
    model.eval()
    with torch.no_grad():
        # Convert genes to indices using vocabulary
        default_idx = list(vocab.values())[0]  # Use first token as default
        gene_indices = torch.tensor([vocab.get(gene, default_idx) for gene in genes])
        
        # Convert expression values to tensor and normalize them to [0, 1] range
        expr_values = torch.tensor(expression_values, dtype=torch.float32)
        expr_values = (expr_values - expr_values.min()) / (expr_values.max() - expr_values.min())
        
        # Create padding mask (no padding in our case)
        padding_mask = torch.zeros(len(genes), dtype=torch.bool)
        
        # Get embeddings using both gene indices and expression values
        output = model._encode(
            src=gene_indices.unsqueeze(0),  # Add batch dimension
            values=expr_values.unsqueeze(0),  # Add batch dimension
            src_key_padding_mask=padding_mask.unsqueeze(0),  # Add batch dimension
        )
        
        # Get the final embeddings (remove batch dimension)
        embeddings = output[0]
    return embeddings

def save_embeddings_by_timepoint(embeddings_dict, output_prefix):
    """Save embeddings for each timepoint."""
    # Save as CSV
    all_timepoints_data = []
    
    for timepoint, (genes, embeddings, expressions) in embeddings_dict.items():
        # Convert embeddings to numpy array
        embeddings_np = embeddings.numpy()
        
        # Create DataFrame for this timepoint
        timepoint_data = pd.DataFrame({
            'timepoint': [timepoint] * len(genes),
            'gene': genes,
            'expression': expressions,
            **{f'dim_{i}': embeddings_np[:, i] for i in range(embeddings_np.shape[1])}
        })
        
        all_timepoints_data.append(timepoint_data)
        
        # Save individual timepoint data
        timepoint_file = f"{output_prefix}_timepoint_{timepoint}.csv"
        timepoint_data.to_csv(timepoint_file, index=False)
        print(f"Embeddings for timepoint {timepoint} saved to {timepoint_file}")
    
    # Save combined data
    combined_data = pd.concat(all_timepoints_data, axis=0, ignore_index=True)
    combined_file = f"{output_prefix}_all_timepoints.csv"
    combined_data.to_csv(combined_file, index=False)
    print(f"\nCombined embeddings for all timepoints saved to {combined_file}")

if __name__ == "__main__":
    # Paths to model files
    BASE_DIR = "/ocean/projects/cis240075p/heidarir/scgpt_model"
    MODEL_PATH = os.path.join(BASE_DIR, "model_files/best_model.pt")
    ARGS_PATH = os.path.join(BASE_DIR, "model_files/args.json")
    VOCAB_PATH = os.path.join(BASE_DIR, "model_files/vocab.json")
    EXPR_PATH = os.path.join(BASE_DIR, "model_files/mean_gene_expression_train_whole.txt")
    
    # Load expression data
    print("Loading expression data...")
    expression_data = load_expression_data(EXPR_PATH)
    print(f"Loaded expression data for {len(expression_data)} timepoints")
    
    # Load the model
    print("Loading model...")
    model, vocab = load_model(MODEL_PATH, ARGS_PATH, VOCAB_PATH)
    print("Model loaded successfully!")
    
    # Dictionary to store embeddings for each timepoint
    embeddings_by_timepoint = {}
    
    # Process each timepoint
    for timepoint, data in expression_data.items():
        print(f"\nProcessing timepoint {timepoint}...")
        genes = data["genes"]
        expressions = data["expressions"]
        
        # Check which genes are in vocabulary
        genes_not_in_vocab = [gene for gene in genes if gene not in vocab]
        if genes_not_in_vocab:
            print(f"Warning: At timepoint {timepoint}, the following genes are not in vocabulary:")
            print(", ".join(genes_not_in_vocab))
            print("They will be represented using a default token.")
        
        # Get embeddings for this timepoint
        embeddings = get_embeddings(model, genes, expressions, vocab)
        print(f"Embeddings shape for timepoint {timepoint}: {embeddings.shape}")
        
        # Store embeddings, genes, and expressions for this timepoint
        embeddings_by_timepoint[timepoint] = (genes, embeddings, expressions)
    
    # Save embeddings for all timepoints
    save_embeddings_by_timepoint(embeddings_by_timepoint, "gene_embeddings_with_expr") 