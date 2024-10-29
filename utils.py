import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from scipy.stats import pearsonr

def get_best_overall_mse_corr(average_results_with_std):
    """
    Find the best overall MSE and correlation aggregate results from the data structure,
    including the standard deviation.
    
    Args:
    - average_results_with_std: The data structure with average and std results.
    
    Returns:
    - best_model_mse: Tuple (model_name, params, mse_mean, mse_std) for the best MSE.
    - best_model_corr: Tuple (model_name, params, corr_mean, corr_std) for the best correlation.
    """
    best_model_mse = None
    best_mse = float('inf')  # Initialize with a large value
    best_mse_std = None

    best_model_corr = None
    best_corr = float('-inf')  # Initialize with a small value
    best_corr_std = None

    # Iterate over the models to find the best MSE and Correlation
    for model_name in ['ridge', 'knn', 'mlp','train_condition']:
        if model_name in average_results_with_std:
            for params, metrics in average_results_with_std[model_name].items():
                mse_mean = metrics['mean']['mse']
                mse_std = metrics['std']['mse']
                corr_mean = metrics['mean'].get('corr', None)
                corr_std = metrics['std'].get('corr', None)

                # Check if this MSE is the best
                if mse_mean < best_mse:
                    best_mse = mse_mean
                    best_mse_std = mse_std
                    best_model_mse = (model_name, params, mse_mean, mse_std)

                # Check if this correlation is the best
                if corr_mean is not None:  # Ensure corr_mean exists
                    if corr_mean > best_corr:
                        best_corr = corr_mean
                        best_corr_std = corr_std
                        best_model_corr = (model_name, params, corr_mean, corr_std)
                else:
                    print(f"Warning: No correlation found for {model_name} with params {params}")

    return best_model_mse, best_model_corr


def get_subset_by_perturbation_type(results, perturbation_type='single'):
    """
    Get a subset of the results based on perturbation type (single or double).

    Args:
    - results: The dictionary containing the results for multiple embeddings.
    - perturbation_type: 'single' or 'double' to filter the results based on perturbation type.

    Returns:
    - filtered_results: A dictionary with the same structure as `results`, but only containing 
                        single or double perturbations as specified.
    """
    # Initialize the filtered results
    filtered_results = {}

    # Iterate over each embedding in the results
    for embedding_name, embedding_results in results.items():
        # Get the ranked_genes for the current embedding
        ranked_genes = embedding_results['ranked_genes']

        # Separate single and double perturbations
        if perturbation_type == 'single':
            filtered_genes = [(gene_name, corr, std_corr) for gene_name, corr, std_corr in ranked_genes if '+' not in gene_name]
        elif perturbation_type == 'double':
            filtered_genes = [(gene_name, corr, std_corr) for gene_name, corr, std_corr in ranked_genes if '+' in gene_name]
        else:
            raise ValueError("perturbation_type must be either 'single' or 'double'")

        # If there are no genes that match the criteria, skip this embedding
        if not filtered_genes:
            continue

        # Convert best_model_mse and best_model_corr tuples to lists for modification
        best_model_mse = list(embedding_results['best_model_mse'])
        best_model_corr = list(embedding_results['best_model_corr'])

        # Update best_model_mse and best_model_corr
        best_model_mse[2] = np.mean([x[2] for x in filtered_genes])  # Mean of std_corr for MSE
        best_model_mse[3] = np.std([x[2] for x in filtered_genes])/np.sqrt(len(filtered_genes)) # Std of std_corr for MSE
        best_model_corr[2] = np.mean([x[1] for x in filtered_genes])  # Mean correlation
        best_model_corr[3] = np.std([x[1] for x in filtered_genes])/np.sqrt(len(filtered_genes))   # Std correlation

        # Convert the modified lists back to tuples
        best_model_mse = tuple(best_model_mse)
        best_model_corr = tuple(best_model_corr)

        # Copy the original results but replace the ranked_genes with the filtered ones
        filtered_results[embedding_name] = {
            'best_model_mse': best_model_mse,
            'best_model_corr': best_model_corr,
            'ranked_genes': filtered_genes
        }

    return filtered_results

 
def get_ranked_genes_by_correlation(accumulated_results, best_model):
    """
    Get a list of genes ranked by average test set correlation for the model with the best aggregate results.

    Args:
    - accumulated_results: The data structure with per-gene accumulated results.
    - best_model: Tuple (model_name, params, corr_mean, corr_std) representing the best model.

    Returns:
    - ranked_genes: List of tuples (gene_name, avg_correlation, avg_mse) sorted by avg_correlation.
    """
    model_name, params, corr_mean, corr_std = best_model  # Unpack the best model

    # Extract per-gene results for the best model
    per_gene_results = {x:accumulated_results['per_gene'][x][model_name][params] for x in accumulated_results['per_gene']}
    
    # Create a list of tuples with gene names and their correlation and mse values for the best model
    gene_corr_list = []
    for gene_name, results in per_gene_results.items():
        # Each element in results is a tuple: (corr, mse, y_pred, y_test)
        corr_value = results[0]  # Extract correlation
        mse_value = results[1]   # Extract mse

        # Append gene name and corresponding correlation and mse values
        gene_corr_list.append((gene_name, corr_value, mse_value))

    # Sort the genes by correlation in descending order
    ranked_genes = sorted(gene_corr_list, key=lambda x: x[1], reverse=True)

    return ranked_genes

def get_gene_predictions(accumulated_results, best_model):
    """
    Generate a dictionary of genes with their corresponding y_pred and y_test based on the best model.
    
    Args:
    - accumulated_results: The data structure with per-gene accumulated results.
    - best_model: Tuple (model_name, params, corr_mean, corr_std) representing the best model.
    
    Returns:
    - gene_predictions: Dictionary with gene names as keys and tuples (y_pred, y_test) as values.
    """
    model_name, params, corr_mean, corr_std = best_model  # Unpack the best model

    # Extract per-gene results for the best model
    per_gene_results = {x:accumulated_results['per_gene'][x][model_name][params] for x in accumulated_results['per_gene']}
    gene_predictions = {}

    for gene_name, results in per_gene_results.items():
        # Each element in results is a tuple: (corr, mse, y_pred, y_test)
        y_pred = results[2]  # Extract predicted values
        y_test = results[3]  # Extract true test values

        # Store the predictions and test values in the dictionary
        gene_predictions[gene_name] = (y_pred, y_test, results[4], results[5])

    return gene_predictions



def run_experiments_with_embeddings(experiment, embedding_pairs, ridge_params=None, knn_params=None, k=5, mlp_epochs=55, use_mlp=False,\
    condition_strategy_list=['mean','median'], mean_baseline = True,output_dir = "./"):
    """
    Run experiments for multiple embeddings and store the results.
    
    Args:
    - experiment: The GenePertExperiment object (with dataset already loaded).
    - embedding_pairs: List of tuples (embedding_name, embedding_path).
    - ridge_params: List of dictionaries with hyperparameters for Ridge regression.
    - knn_params: List of dictionaries with hyperparameters for KNN regression.
    - k: Number of folds for cross-validation (default 5).
    
    Returns:
    - results_comparison: A dictionary with best model MSE, best model correlation, 
                          ranked genes, and gene predictions for each embedding.
    """
    results_comparison = {}

    # Iterate over the embedding pairs
    for embedding_name, embedding_path in embedding_pairs:
        print(f"Running experiment for embedding: {embedding_name}")

        # Load the embeddings from the specified path
        with open(embedding_path, "rb") as fp:
            embeddings = pickle.load(fp)
        
        # Update the embeddings in the experiment object
        experiment.embeddings = embeddings

        # Run k-fold cross-validation with the specified parameters
        results_with_kfold, accumulated_results = experiment.run_kfold_experiments(
            ridge_params=ridge_params,
            knn_params=knn_params,
            k=k,
            mlp_epochs=mlp_epochs,
            use_mlp=use_mlp,
            condition_strategy_list=condition_strategy_list,
            output_dir=output_dir,
            mean_baseline=mean_baseline
        )

        # Get the best overall MSE and correlation
        best_model_mse, best_model_corr = get_best_overall_mse_corr(results_with_kfold)

        if best_model_corr is None:
            print(f"Warning: No valid correlation model found for {embedding_name}. Skipping...")
            continue  # Skip to the next embedding if no valid model is found

        # Get the ranked genes by correlation for the model with the best correlation
        # genes info should be obtained from 
        ranked_genes = get_ranked_genes_by_correlation(accumulated_results, best_model_corr)

        # Get the gene predictions (y_pred and y_test) for the best model
        gene_predictions = get_gene_predictions(accumulated_results, best_model_corr)

        # Save the results for this embedding
        results_comparison[embedding_name] = {
            'best_model_mse': best_model_mse,  # Save best MSE results
            'best_model_corr': best_model_corr,  # Save best correlation results
            'ranked_genes': ranked_genes,  # Save the ranked genes
            'gene_predictions': gene_predictions  # Save gene predictions (y_pred and y_test)
        }

        if "train_condition" not in results_comparison:
            best_model_mse, best_model_corr = get_best_overall_mse_corr({'train_condition':results_with_kfold["train_condition"]})
            ranked_genes = get_ranked_genes_by_correlation(accumulated_results, best_model_corr)
            # Get the gene predictions (y_pred and y_test) for the best model
            gene_predictions = get_gene_predictions(accumulated_results, best_model_corr)
            results_comparison['train_condition'] = {
            'best_model_mse': best_model_mse,  # Save best MSE results
            'best_model_corr': best_model_corr,  # Save best correlation results
            'ranked_genes': ranked_genes,  # Save the ranked genes
            'gene_predictions': gene_predictions  # Save gene predictions (y_pred and y_test)
        }

        # Output progress
        print(f"Finished experiment for embedding: {embedding_name}")
        print(f"Best MSE for {embedding_name}: {best_model_mse}")
        print(f"Best Correlation for {embedding_name}: {best_model_corr}")
    
    return results_comparison

def plot_mse_corr_comparison(results_comparison, dataset_name, axis_a_lift = 0.02, axis_b_lift=0.01, print_table = True):
    """
    Plot the mean and standard deviation for MSE (scaled) and correlation across embeddings with improved aesthetics.
    
    Args:
    - results_comparison: Dictionary of results for each embedding.
    """
    # Scaling factor for MSE
    scale_factor = 100
    
    embeddings = []
    mse_means, mse_stds = [], []
    corr_means, corr_stds = [], []  # Initialize both as empty lists

    # Extract data from the results_comparison
    for embedding_name, result in results_comparison.items():
        embeddings.append(embedding_name)
        best_model_corr = result['best_model_corr']
        best_model_mse = result['best_model_mse']

        # Extract MSE and Correlation mean and std, scaling MSE by 100
        mse_means.append(best_model_mse[2] * scale_factor)  # mean MSE
        mse_stds.append(best_model_mse[3] * scale_factor)   # std MSE
        corr_means.append(best_model_corr[2])  # mean correlation
        corr_stds.append(best_model_corr[3])   # std correlation

    # Convert lists to numpy arrays for easier plotting
    mse_means = np.array(mse_means)
    mse_stds = np.array(mse_stds)
    corr_means = np.array(corr_means)
    corr_stds = np.array(corr_stds)

    # Create bar plots for MSE and Correlation
    x = np.arange(len(embeddings))  # Label locations

    # Set font to Helvetica    
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # Define colors and styles similar to the attached figure
    colors = sns.color_palette("Set3", len(embeddings))  # Using Set3 for diverse colors
    bar_width = 0.6  # Thinner bars

    # Bar plot for MSE (scaled)
    mse_bars = ax[0].bar(x, mse_means, yerr=mse_stds, width=bar_width, color=colors, align='center', alpha=0.9, 
              ecolor='black', capsize=10, edgecolor='none', linewidth=0.5)
    ax[0].set_ylabel('Mean RMSE (x10-2)', fontsize=14)
    ax[0].set_title('Best 5-fold CV MSE across Embedding Models', fontsize=16)

    # Move data labels below the bars and rotate vertically for MSE bars
    for i, bar in enumerate(mse_bars):
        ax[0].text(bar.get_x() + bar.get_width() / 2,axis_a_lift, 
                   f"{mse_means[i]:.3f}\n±{mse_stds[i]:.3f}", ha='center', fontsize=10, color='black', alpha=0.8, rotation=90)

    # Bar plot for Correlation
    corr_bars = ax[1].bar(x, corr_means, yerr=corr_stds, width=bar_width, color=colors, align='center', alpha=0.9, 
              ecolor='black', capsize=10, edgecolor='none', linewidth=0.5)
    ax[1].set_ylabel('Mean Correlation', fontsize=14)
    ax[1].set_title('Best 5-fold CV Correlation across Embedding Models', fontsize=16)

    # Move data labels below the bars and rotate vertically for Correlation bars
    for i, bar in enumerate(corr_bars):
        ax[1].text(bar.get_x() + bar.get_width() / 2, axis_b_lift, 
                   f"{corr_means[i]:.3f}\n±{corr_stds[i]:.3f}", ha='center', fontsize=10, color='black', alpha=0.8, rotation=90)

    # Remove x-axis labels and use a shared legend instead
    ax[0].set_xticks([])
    ax[1].set_xticks([])

    plt.suptitle(f"Performance on {dataset_name}", fontsize=18, fontweight='bold')

    # Create custom legend using Patch objects
    legend_patches = [Patch(facecolor=colors[i], label=embeddings[i]) for i in range(len(embeddings))]

    # Add a shared legend below the plot
    fig.legend(handles=legend_patches, loc='lower center', ncol=len(embeddings)//2, bbox_to_anchor=(0.5, -0.15), fontsize=12, frameon=False)

    # Set white background
    fig.patch.set_facecolor('white')

    # Adjust layout for better aesthetics
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leaves space for the legend
    plt.show()

    if print_table:
        # Print RMSE and Correlation results in LaTeX table format with dataset_name as the first column
        print("\\begin{table}[ht]\n\\centering\n\\begin{tabular}{lccc}")
        print("\\toprule")
        print(f"Dataset & Embedding & RMSE (x$10^{{-2}}$) & Correlation \\\\")
        print("\\midrule")
        for i, embedding in enumerate(embeddings):
            mse_line = f"{mse_means[i]:.2f} $\\pm$ {mse_stds[i]:.2f}"
            corr_line = f"{corr_means[i]:.2f} $\\pm$ {corr_stds[i]:.2f}"
            if i == 0:
                print(f"{dataset_name} & {embedding} & {mse_line} & {corr_line} \\\\")
            else:
                print(f" & {embedding} & {mse_line} & {corr_line} \\\\")
        print("\\bottomrule")
        print(f"\\end{{tabular}}\n\\caption{{Mean and standard deviation of RMSE (scaled) and correlation for {dataset_name}.}}\n\\end{{table}}")


def compare_embedding_correlations(results, output_dir='./'):
    """
    Compare correlation values between pairs of embeddings using scatter plots.
    
    Args:
    - results: A dictionary containing results for multiple embeddings, where each embedding 
               has a 'ranked_genes' key with (gene_name, correlation, std_corr).
    
    Returns:
    - None (outputs scatter plots and correlation values for each pair of embeddings).
    """
    embedding_names = list(results.keys())
    
    # Iterate over all pairs of embeddings
    for i in range(len(embedding_names)):
        for j in range(i+1, len(embedding_names)):
            emb_name_1 = embedding_names[i]
            emb_name_2 = embedding_names[j]
            
            # Get the ranked genes for both embeddings
            ranked_genes_1 = results[emb_name_1]['ranked_genes']
            ranked_genes_2 = results[emb_name_2]['ranked_genes']
            
            # Create dictionaries for gene correlations in each embedding
            gene_corr_1 = {gene_name: corr for gene_name, corr, _ in ranked_genes_1}
            gene_corr_2 = {gene_name: corr for gene_name, corr, _ in ranked_genes_2}
            
            # Find common genes between the two embeddings
            common_genes = set(gene_corr_1.keys()).intersection(gene_corr_2.keys())
            
            # Extract correlation values for the common genes
            corr_values_1 = [gene_corr_1[gene] for gene in common_genes]
            corr_values_2 = [gene_corr_2[gene] for gene in common_genes]
            
            # Compute the Pearson correlation between the two embeddings
            pearson_corr, _ = pearsonr(corr_values_1, corr_values_2)
            print(f"Pearson correlation between {emb_name_1} and {emb_name_2}: {pearson_corr:.3f}")
            
            # Plot the scatter plot for the two embeddings
            plt.figure(figsize=(6, 6))
            plt.scatter(corr_values_1, corr_values_2, alpha=0.65)  # Set transparency to 0.65
            plt.plot([min(corr_values_1), max(corr_values_1)], [min(corr_values_1), max(corr_values_1)], 'r--')  # y=x line
            
            # Set plot labels and title
            plt.xlabel(f'{emb_name_1} Correlation', fontsize=12)
            plt.ylabel(f'{emb_name_2} Correlation', fontsize=12)
            plt.title(f'Correlation Comparison\n{emb_name_1} vs {emb_name_2}', fontsize=14)
            
            # Display the plot
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/comparing_{emb_name_1}_vs_{emb_name_2}.svg", dpi=350)

