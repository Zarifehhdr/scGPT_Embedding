import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

def load_embeddings(file_prefix="gene_embeddings_with_expr"):
    """Load all embeddings from the CSV files."""
    return pd.read_csv(f"{file_prefix}_all_timepoints.csv")

def get_embedding_matrix(df):
    """Extract embedding matrix from dataframe."""
    embedding_cols = [col for col in df.columns if col.startswith('dim_')]
    return df[embedding_cols].values

def find_similar_genes(df, gene_name, timepoint, n_similar=5):
    """Find most similar genes based on embedding cosine similarity."""
    # Filter data for the specific timepoint
    df_timepoint = df[df['timepoint'] == timepoint].reset_index(drop=True)
    
    # Get embeddings for this timepoint
    embeddings = get_embedding_matrix(df_timepoint)
    
    # Get the query gene's embedding
    query_idx = df_timepoint[df_timepoint['gene'] == gene_name].index[0]
    query_embedding = embeddings[query_idx].reshape(1, -1)
    
    # Calculate cosine similarities
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    
    # Get indices of top similar genes (excluding the query gene itself)
    similar_indices = np.argsort(similarities)[::-1][1:n_similar+1]
    
    # Create result dataframe
    similar_genes = pd.DataFrame({
        'Gene': df_timepoint['gene'].iloc[similar_indices],
        'Similarity': similarities[similar_indices],
        'Expression': df_timepoint['expression'].iloc[similar_indices]
    })
    
    return similar_genes

def plot_similarity_heatmap(df, genes_of_interest, output_file="gene_similarities.png"):
    """Plot similarity heatmap for specified genes across all timepoints."""
    # Get unique timepoints
    timepoints = sorted(df['timepoint'].unique())
    
    # Create a figure with subplots for each timepoint
    n_timepoints = len(timepoints)
    fig, axes = plt.subplots(1, n_timepoints, figsize=(5*n_timepoints, 4))
    if n_timepoints == 1:
        axes = [axes]
    
    for ax, timepoint in zip(axes, timepoints):
        # Filter data for this timepoint
        df_time = df[df['timepoint'] == timepoint]
        
        # Get embeddings for genes of interest
        genes_data = df_time[df_time['gene'].isin(genes_of_interest)]
        embeddings = get_embedding_matrix(genes_data)
        
        # Calculate similarity matrix
        sim_matrix = cosine_similarity(embeddings)
        
        # Create heatmap
        sns.heatmap(sim_matrix, annot=True, cmap='viridis', vmin=0, vmax=1,
                    xticklabels=genes_data['gene'], yticklabels=genes_data['gene'],
                    ax=ax)
        ax.set_title(f'Time point: {timepoint}')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def find_most_similar_pairs(df, n_pairs=10):
    """Find the most similar gene pairs across all timepoints."""
    results = []
    
    for timepoint in df['timepoint'].unique():
        # Filter data for this timepoint
        df_time = df[df['timepoint'] == timepoint]
        
        # Get embeddings
        embeddings = get_embedding_matrix(df_time)
        
        # Calculate similarity matrix
        sim_matrix = cosine_similarity(embeddings)
        
        # Set diagonal to -1 to exclude self-similarities
        np.fill_diagonal(sim_matrix, -1)
        
        # Find top pairs
        for _ in range(n_pairs):
            # Get indices of maximum similarity
            i, j = np.unravel_index(sim_matrix.argmax(), sim_matrix.shape)
            similarity = sim_matrix[i, j]
            
            # Add to results
            results.append({
                'Timepoint': timepoint,
                'Gene1': df_time['gene'].iloc[i],
                'Gene2': df_time['gene'].iloc[j],
                'Similarity': similarity,
                'Expression1': df_time['expression'].iloc[i],
                'Expression2': df_time['expression'].iloc[j]
            })
            
            # Set this pair's similarity to -1 to find next highest
            sim_matrix[i, j] = -1
            sim_matrix[j, i] = -1
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    # Load embeddings
    print("Loading embeddings...")
    df = load_embeddings()
    
    # 1. Find most similar pairs across all timepoints
    print("\nFinding most similar gene pairs...")
    similar_pairs = find_most_similar_pairs(df, n_pairs=5)
    print("\nMost similar gene pairs across all timepoints:")
    print(similar_pairs.to_string(index=False))
    
    # 2. For each timepoint, find similar genes for the gene with highest expression
    print("\nFinding similar genes for highest expressed genes at each timepoint...")
    for timepoint in sorted(df['timepoint'].unique()):
        df_time = df[df['timepoint'] == timepoint].reset_index(drop=True)
        top_gene = df_time.loc[df_time['expression'].idxmax(), 'gene']
        
        print(f"\nTimepoint {timepoint}")
        print(f"Most similar genes to {top_gene} (highest expressed gene):")
        similar_genes = find_similar_genes(df, top_gene, timepoint)
        print(similar_genes.to_string(index=False))
        
        # Plot similarity heatmap for this gene and its similar genes
        genes_to_plot = [top_gene] + similar_genes['Gene'].tolist()
        plot_similarity_heatmap(df_time, genes_to_plot, 
                              f"gene_similarities_t{timepoint}.png")
    
    # Save results to files
    similar_pairs.to_csv("most_similar_gene_pairs.csv", index=False)
    print("\nResults have been saved to 'most_similar_gene_pairs.csv' and similarity heatmaps have been generated.") 