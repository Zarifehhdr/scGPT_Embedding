import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

def load_embeddings(file_prefix="gene_embeddings_with_expr"):
    """Load all embeddings from the CSV files."""
    return pd.read_csv(f"{file_prefix}_all_timepoints.csv")

def plot_expression_heatmap(df, output_file="expression_heatmap.png"):
    """Create a heatmap of gene expressions across time."""
    # Pivot the data to create a matrix of time points x genes
    pivot_df = df.pivot(index='timepoint', columns='gene', values='expression')
    
    plt.figure(figsize=(15, 8))
    sns.heatmap(pivot_df, cmap='viridis', center=0)
    plt.title('Gene Expression Heatmap Across Time Points')
    plt.ylabel('Time Point (days)')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def plot_dimension_reduction(df, method='pca', n_components=2, output_file=None):
    """Create dimension reduction plots (PCA, t-SNE, or UMAP)."""
    # Get embedding dimensions
    embedding_cols = [col for col in df.columns if col.startswith('dim_')]
    embeddings = df[embedding_cols].values
    
    # Perform dimension reduction
    if method.lower() == 'pca':
        reducer = PCA(n_components=n_components)
    elif method.lower() == 'tsne':
        reducer = TSNE(n_components=n_components, random_state=42)
    elif method.lower() == 'umap':
        reducer = umap.UMAP(n_components=n_components, random_state=42)
    
    reduced_embeddings = reducer.fit_transform(embeddings)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1],
                         c=df['timepoint'], cmap='viridis',
                         s=100, alpha=0.6)
    
    # Add labels for each point
    for i, gene in enumerate(df['gene']):
        plt.annotate(gene, (reduced_embeddings[i, 0], reduced_embeddings[i, 1]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.colorbar(scatter, label='Time (days)')
    plt.title(f'Gene Embeddings Visualization using {method.upper()}')
    plt.xlabel(f'{method.upper()} Component 1')
    plt.ylabel(f'{method.upper()} Component 2')
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def plot_gene_trajectories(df, genes_to_plot=None, output_file="gene_trajectories.png"):
    """Plot trajectories of specific genes across time using PCA."""
    # Get embedding dimensions
    embedding_cols = [col for col in df.columns if col.startswith('dim_')]
    embeddings = df[embedding_cols].values
    
    # Perform PCA
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)
    
    # Add reduced dimensions to dataframe
    df['PCA1'] = reduced_embeddings[:, 0]
    df['PCA2'] = reduced_embeddings[:, 1]
    
    # If no specific genes provided, select top 5 genes with highest variance in expression
    if genes_to_plot is None:
        expr_var = df.groupby('gene')['expression'].var().sort_values(ascending=False)
        genes_to_plot = expr_var.head(5).index.tolist()
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Plot trajectories for selected genes
    for gene in genes_to_plot:
        gene_data = df[df['gene'] == gene].sort_values('timepoint')
        plt.plot(gene_data['PCA1'], gene_data['PCA2'], 'o-', label=gene, linewidth=2, markersize=8)
        
        # Add time labels
        for _, row in gene_data.iterrows():
            plt.annotate(f"t={int(row['timepoint'])}",
                        (row['PCA1'], row['PCA2']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.title('Gene Expression Trajectories Over Time')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def plot_expression_vs_embedding(df, output_file="expr_vs_embedding.png"):
    """Plot relationship between expression values and embedding dimensions."""
    # Get embedding dimensions
    embedding_cols = [col for col in df.columns if col.startswith('dim_')]
    
    # Calculate embedding magnitudes
    embedding_magnitudes = np.sqrt(np.sum(df[embedding_cols].values ** 2, axis=1))
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot with points colored by time
    scatter = plt.scatter(df['expression'], embedding_magnitudes, 
                         c=df['timepoint'], cmap='viridis',
                         alpha=0.6, s=100)
    
    # Add gene labels for points with highest expression or embedding magnitude
    n_labels = 5  # Number of points to label
    indices_to_label = list(df['expression'].nlargest(n_labels).index) + \
                      list(pd.Series(embedding_magnitudes).nlargest(n_labels).index)
    
    for idx in set(indices_to_label):  # use set to remove duplicates
        plt.annotate(f"{df['gene'].iloc[idx]}\nt={df['timepoint'].iloc[idx]}",
                    (df['expression'].iloc[idx], embedding_magnitudes[idx]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Add trend line
    z = np.polyfit(df['expression'], embedding_magnitudes, 1)
    p = np.poly1d(z)
    plt.plot(df['expression'], p(df['expression']), "r--", alpha=0.8,
             label=f'Trend line (R² = {np.corrcoef(df["expression"], embedding_magnitudes)[0,1]**2:.3f})')
    
    plt.colorbar(scatter, label='Time (days)')
    plt.xlabel('Expression Value')
    plt.ylabel('Embedding Magnitude')
    plt.title('Relationship between Gene Expression and Embedding Magnitude')
    plt.legend()
    
    # Add text with correlation information
    corr = np.corrcoef(df['expression'], embedding_magnitudes)[0,1]
    plt.text(0.02, 0.98, f'Correlation: {corr:.3f}',
             transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Load the embeddings
    print("Loading embeddings...")
    df = load_embeddings()
    
    print("Generating plots...")
    
    # 1. Expression heatmap
    print("Creating expression heatmap...")
    plot_expression_heatmap(df)
    
    # 2. Dimension reduction plots
    print("Creating PCA plot...")
    plot_dimension_reduction(df, method='pca', output_file='pca_visualization.png')
    
    print("Creating t-SNE plot...")
    plot_dimension_reduction(df, method='tsne', output_file='tsne_visualization.png')
    
    print("Creating UMAP plot...")
    plot_dimension_reduction(df, method='umap', output_file='umap_visualization.png')
    
    # 3. Gene trajectories
    print("Creating gene trajectories plot...")
    # Select genes with highest expression variance
    expr_var = df.groupby('gene')['expression'].var().sort_values(ascending=False)
    top_genes = expr_var.head(5).index.tolist()
    plot_gene_trajectories(df, genes_to_plot=top_genes)
    
    # 4. Expression vs Embedding magnitude
    print("Creating expression vs embedding magnitude plot...")
    plot_expression_vs_embedding(df)
    
    print("All plots have been generated!") 