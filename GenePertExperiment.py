import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
import tqdm
from sklearn.model_selection import KFold
from matplotlib.patches import Patch
import scanpy as sc
from sklearn.metrics.pairwise import cosine_similarity
import json
import os

# we might need to make the distance more efficient
def calculate_distances(X_train, X_test):
    closest_distances = []
    avg_top10_distances = []
    
    for i in range(X_test.shape[0]):
        similarities = cosine_similarity(X_test[i].reshape(1, -1), X_train)[0]
        distances = 1 - similarities
        
        # Sort distances to get closest ones
        sorted_distances = np.sort(distances)
        
        # Get closest distance and average of top 10 closest distances
        closest_distances.append(sorted_distances[0])
        avg_top10_distances.append(np.mean(sorted_distances[:10]))

    return {
        'closest_distances': np.array(closest_distances),
        'avg_top10_distances': np.array(avg_top10_distances)
    }

# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=128):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x) # we know its non-negative 
        return x

class TrainConditionModel:
    def __init__(self, strategy='mean'):
        """
        Initialize the TrainConditionModel with a strategy: 'mean' or 'median'
        """
        if strategy not in ['mean', 'median']:
            raise ValueError("Strategy must be 'mean' or 'median'")
        self.strategy = strategy
        self.constant_value = None

    def fit(self, X_train, y_train):
        """
        Fit the model by calculating the mean or median of y_train.
        X_train is not used in this model, but it is accepted for compatibility with other models.
        """
        if self.strategy == 'mean':
            self.constant_value = np.mean(y_train, axis=0)
        elif self.strategy == 'median':
            self.constant_value = np.median(y_train, axis=0)

    def predict(self, X_test):
        """
        Predict based on the constant value (mean or median) from y_train.
        X_test is not used for prediction, but it is accepted for compatibility with other models.
        """
        # Repeat the constant value for the same number of rows as X_test
        return np.tile(self.constant_value, (X_test.shape[0], 1))

def clean_condition(condition):
    return condition.replace('+ctrl', '').replace('ctrl+', '').strip()

def populate_dicts(adata_subset, mean_dict):
    for condition in adata_subset.obs['condition'].unique():
        condition_mask = adata_subset.obs['condition'] == condition
        condition_data = adata_subset[condition_mask].X
        clean_cond = clean_condition(condition)
        mean_dict[clean_cond] = np.mean(condition_data, axis=0)

class GenePertExperiment:
    def __init__(self, embeddings):
        """
        Initialize the experiment with preloaded embeddings.
        """
        self.embeddings = embeddings
        self.mean_expression = None

    def load_dataset(self, dataset_path):
        """
        Load the dataset (h5ad format) and store it.
        """
        self.adata = sc.read_h5ad(dataset_path)
        self.mean_expression = self.get_mean_control()
        
    def clean_condition(self, condition):
        return condition.replace('+ctrl', '').replace('ctrl+', '').strip()
    
    def get_mean_control(self, control_label='ctrl'):
        """
        Get mean control expression
        """
        mean_ctrl_exp = np.array(self.adata[self.adata.obs['condition'] == control_label].to_df().mean())
        return mean_ctrl_exp

    def evaluate_performance_rowwise(self, y_true, y_pred, mean_expression = None):
        """
        Evaluate performance metrics row-wise (for each condition).
        """
        if mean_expression is None:
            mean_expression = self.mean_expression
        y_true_centered = y_true - self.mean_expression
        y_pred_centered = y_pred - self.mean_expression
        n_rows = y_true.shape[0]
        mse_list, mae_list, corr_list = [], [], []

        for i in range(n_rows):
            # print('input',y_true_centered[i])
            # print('output',y_pred_centered[i])
            mse = np.sqrt(mean_squared_error(y_true_centered[i], y_pred_centered[i]))
            mae = mean_absolute_error(y_true_centered[i], y_pred_centered[i])
            corr = pearsonr(y_true_centered[i], y_pred_centered[i])[0]
            mse_list.append(mse)
            mae_list.append(mae)
            corr_list.append(corr)

        return mse_list, mae_list, corr_list

    def populate_X_y(self, mean_dict, X, y, embedding_size, interaction=False):
        """
        Populate the input (X) and target (y) matrices based on gene embeddings and mean values.
        
        Args:
        - mean_dict: A dictionary of gene names and mean values.
        - X: The feature matrix to be populated.
        - y: The target matrix to be populated.
        - embedding_size: Size of the embedding vector.
        - interaction: Whether to compute interaction terms (default False).
        
        Returns:
        - gene_name_X_map: A list mapping gene names to the corresponding features.
        """
        gene_name_X_map = []

        for gene_name, mean_value in mean_dict.items():
            # Infer whether the perturbation is single or multiple based on the number of genes
            genes = gene_name.split('+')  # Split by '+' to handle multiple perturbations

            if len(genes) == 1:  # Single perturbation
                single_gene_name = genes[0]
                if single_gene_name in self.embeddings:
                    X.append(self.embeddings[single_gene_name]/np.linalg.norm(self.embeddings[single_gene_name]))
                    gene_name_X_map.append(single_gene_name)
                else:
                    # Generate a random vector and normalize it to have an ell 2 norm of 1
                    random_vector = np.random.randn(embedding_size)
                    normalized_vector = random_vector / np.linalg.norm(random_vector)
                    X.append(normalized_vector)
            else:  # Multiple perturbations
                valid_genes = [g for g in genes if g in self.embeddings]

                if len(valid_genes) == len(genes):  # All genes are found in embeddings
                    embeddings = [self.embeddings[g] for g in valid_genes]
                    
                    # Sum the embeddings
                    combined_embedding = np.sum(embeddings, axis=0)
                    combined_embedding = combined_embedding / np.linalg.norm(combined_embedding)

                    if interaction:
                        # Add elementwise interaction terms (a * b * c ... if there are multiple genes)
                        interaction_embedding = np.ones_like(embeddings[0])
                        for embedding in embeddings:
                            interaction_embedding *= embedding  # Elementwise multiplication
                        
                        # Concatenate summed embedding and interaction terms
                        concatenated_features = np.concatenate([combined_embedding, interaction_embedding])
                        X.append(concatenated_features)  # Append the concatenated features
                    else:
                        # Just append the summed embedding
                        X.append(combined_embedding)

                    # Append the combined gene name
                    gene_name_X_map.append(gene_name)
                else:
                    # If one or more genes are missing, generate random vector
                    print('genes missing', genes)
                    random_vector = np.random.randn(embedding_size)
                    normalized_vector = random_vector / np.linalg.norm(random_vector)
                    X.append(normalized_vector)

            # Append the corresponding mean value to y
            y.append(np.asarray(mean_value).flatten())

        return gene_name_X_map

    def run_experiment_with_conditions(self, train_conditions, test_conditions, condition_column = "condition", \
        ridge_params=None, knn_params=None, hidden_size=128, mlp_epochs=10, val_split=0.2, use_mlp=False,\
        condition_strategy_list = ['mean','median'], mean_baseline=True):
        """
        Function to run the experiment using provided train/test conditions directly, including MLP training with validation split.

        Args:
        - train_conditions: list of conditions to use for training.
        - test_conditions: list of conditions to use for testing.
        - ridge_params: dictionary of hyperparameters for Ridge regression.
        - knn_params: dictionary of hyperparameters for KNN regression.
        - hidden_size: hidden layer size for MLP (default: 128).
        - mlp_epochs: number of training epochs for MLP (default: 100).
        - val_split: fraction of training data to use for validation (default: 0.2).
        - use_mlp: Boolean to toggle the use of MLP model (default: True)

        Returns:
        - results: A dictionary with results for this experiment.
        """
        results = {}
        embedding_size = len(next(iter(self.embeddings.values())))

        # Set default parameters if not provided
        if ridge_params is None:
            ridge_params = [{'alpha': 0.1}, {'alpha': 1.0}, {'alpha': 10.0}]
        if knn_params is None:
            knn_params = [{'n_neighbors': 1}, {'n_neighbors': 5}, {'n_neighbors': 10}]

        X_train, y_train, X_test, y_test = [], [], [], []

        # Create masks for training and test conditions
        train_mask = self.adata.obs[condition_column].isin(train_conditions)
        test_mask = self.adata.obs[condition_column].isin(test_conditions)

        adata_train = self.adata[train_mask]
        adata_test = self.adata[test_mask]

        mean_dict_train, mean_dict_test = {}, {}

        # Populate training and test sets
        populate_dicts(adata_train, mean_dict_train)
        populate_dicts(adata_test, mean_dict_test)

        # Populate training and test sets and return gene-name-to-X mapping
        train_gene_name_X_map = self.populate_X_y(mean_dict_train, X_train, y_train, embedding_size)
        test_gene_name_X_map = self.populate_X_y(mean_dict_test, X_test, y_test, embedding_size)

        # Convert lists to NumPy arrays
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_test, y_test = np.array(X_test), np.array(y_test)
        distance_results = calculate_distances(X_train, X_test)

        # Split the training set into train_mlp and validation_mlp
        X_train_mlp, X_val_mlp, y_train_mlp, y_val_mlp = train_test_split(X_train, y_train, test_size=val_split, random_state=42)

        # Shape compatibility check
        # print(X_train.shape)
        if X_train.shape[1] != X_test.shape[1] or y_train.shape[1] != y_test.shape[1]:
            raise ValueError("Shape mismatch between training and testing sets.")

            # Store results for this run
        run_results = {'ridge': {}, 'knn': {}, 'mlp': {}, 'train_condition': {}}
        results_per_gene = {}

        # TrainConditionModel evaluation
        if mean_baseline:
            for condition_strategy in condition_strategy_list:
                condition_model = TrainConditionModel(strategy=condition_strategy)
                condition_model.fit(X_train, y_train)
                y_pred_condition = condition_model.predict(X_test)

                # Evaluate row-wise (gene-wise) performance for TrainConditionModel
                mse_condition, mae_condition, corr_condition = self.evaluate_performance_rowwise(y_test, y_pred_condition)
                run_results['train_condition'][condition_strategy] = {'mse': np.mean(mse_condition), 'mae': np.mean(mae_condition), 'corr': np.mean(corr_condition)}

                # Save per-gene performance results including y_pred_condition
                for i, gene_name in enumerate(test_gene_name_X_map):
                    if gene_name not in results_per_gene:
                        results_per_gene[gene_name] = {'ridge': {}, 'knn': {}, 'mlp': {}, 'train_condition': {}}
                    results_per_gene[gene_name]['train_condition'][condition_strategy] = (corr_condition[i], mse_condition[i], y_pred_condition[i], y_test[i],
                         distance_results['closest_distances'][i],distance_results['avg_top10_distances'][i])

        # Ridge Regression evaluation
        for ridge_param in ridge_params:
            ridge_model = Ridge(**ridge_param)
            ridge_model.fit(X_train, y_train)
            y_pred_ridge = ridge_model.predict(X_test)
            # print(y_pred_ridge)
            # print(y_test)

            # Evaluate row-wise (gene-wise) performance
            mse_ridge, mae_ridge, corr_ridge = self.evaluate_performance_rowwise(y_test, y_pred_ridge)
            run_results['ridge'][tuple(ridge_param.items())] = {'mse': np.mean(mse_ridge), 'mae': np.mean(mae_ridge), 'corr': np.mean(corr_ridge)}

            # Save per-gene performance results including y_pred_ridge using `test_gene_name_X_map`
            for i, gene_name in enumerate(test_gene_name_X_map):
                if gene_name not in results_per_gene:
                    results_per_gene[gene_name] = {'ridge': {}, 'knn': {}, 'mlp': {}}
                results_per_gene[gene_name]['ridge'][tuple(ridge_param.items())] = (corr_ridge[i], mse_ridge[i], y_pred_ridge[i], y_test[i], \
                    distance_results['closest_distances'][i],distance_results['avg_top10_distances'][i])

        # KNN evaluation
        for knn_param in knn_params:
            knn_model = KNeighborsRegressor(**knn_param)
            knn_model.fit(X_train, y_train)
            y_pred_knn = knn_model.predict(X_test)

            # Evaluate row-wise (gene-wise) performance
            mse_knn, mae_knn, corr_knn = self.evaluate_performance_rowwise(y_test, y_pred_knn)
            run_results['knn'][tuple(knn_param.items())] = {'mse': np.mean(mse_knn), 'mae': np.mean(mae_knn), 'corr': np.mean(corr_knn)}

            # Save per-gene performance results including y_pred_knn using `test_gene_name_X_map`
            for i, gene_name in enumerate(test_gene_name_X_map):
                if gene_name not in results_per_gene:
                    results_per_gene[gene_name] = {'ridge': {}, 'knn': {}, 'mlp': {}}
                results_per_gene[gene_name]['knn'][tuple(knn_param.items())] = (corr_knn[i], mse_knn[i], y_pred_knn[i], y_test[i],\
                    distance_results['closest_distances'][i],distance_results['avg_top10_distances'][i])

        # MLP evaluation (if use_mlp is True)
        if use_mlp:
            input_dim = X_train_mlp.shape[1]
            output_dim = y_train_mlp.shape[1]

            # Convert NumPy arrays to PyTorch tensors
            X_train_tensor = torch.tensor(X_train_mlp, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train_mlp, dtype=torch.float32)
            X_val_tensor = torch.tensor(X_val_mlp, dtype=torch.float32)
            y_val_tensor = torch.tensor(y_val_mlp, dtype=torch.float32)
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
            y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
            
            # generate a new MLP model each time
            mlp_model = MLP(input_dim=input_dim, hidden_size=hidden_size, output_dim=output_dim)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(mlp_model.parameters(), lr=0.001)

            # Training loop with validation loss
            for epoch in range(mlp_epochs):
                mlp_model.train()
                optimizer.zero_grad()

                # Forward pass for training data
                outputs_train = mlp_model(X_train_tensor)
                train_loss = criterion(outputs_train, y_train_tensor)

                # Backward pass and optimization
                train_loss.backward()
                optimizer.step()

                # Validation loss (model in evaluation mode)
                mlp_model.eval()
                with torch.no_grad():
                    outputs_val = mlp_model(X_val_tensor)
                    val_loss = criterion(outputs_val, y_val_tensor)

            # MLP evaluation on test data
            mlp_model.eval()
            with torch.no_grad():
                y_pred_mlp = mlp_model(X_test_tensor).numpy()

            mse_mlp, mae_mlp, corr_mlp = self.evaluate_performance_rowwise(y_test, y_pred_mlp)
            run_results['mlp'][f"mlp_epochs{mlp_epochs}_hidden_size{hidden_size}"] = {'mse': np.mean(mse_mlp), 'mae': np.mean(mae_mlp), 'corr': np.mean(corr_mlp)}

            # Save per-gene performance results using `test_gene_name_X_map`
            for i, gene_name in enumerate(test_gene_name_X_map):
                if gene_name not in results_per_gene:
                    results_per_gene[gene_name] = {'ridge': {}, 'knn': {}, 'mlp': {}}
                results_per_gene[gene_name]['mlp'][f"mlp_epochs{mlp_epochs}_hidden_size{hidden_size}"] = (corr_mlp[i], mse_mlp[i], y_pred_mlp[i], y_test[i])
        else:
            y_pred_mlp = None  # If MLP is not used, set predictions to None

        # Return the final results including raw predictions and true test values
        results = {
            'aggregate': run_results,
            'per_gene': results_per_gene,
        }

        return results

    def run_experiment_with_adata(self, adata_train, adata_test, ridge_params=None, knn_params=None):
        """
        Function to run the experiment using provided training and testing data directly.
        
        Args:
        - adata_train: AnnData object for training data.
        - adata_test: AnnData object for testing data.
        - ridge_params: dictionary of hyperparameters for Ridge regression.
        - knn_params: dictionary of hyperparameters for KNN regression.

        Returns:
        - results: A dictionary with results for this experiment.
        """
        results = {}
        embedding_size = len(next(iter(self.embeddings.values())))

        # Set default parameters if not provided
        if ridge_params is None:
            ridge_params = [{'alpha': 0.1}, {'alpha': 1.0}, {'alpha': 10.0}]
        if knn_params is None:
            knn_params = [{'n_neighbors': 1}, {'n_neighbors': 5}, {'n_neighbors': 10}]

        X_train, y_train, X_test, y_test = [], [], [], []

        mean_dict_train, mean_dict_test = {}, {}

        # Populate training and test sets
        populate_dicts(adata_train, mean_dict_train)
        populate_dicts(adata_test, mean_dict_test)

        # Populate training and test sets and return gene-name-to-X mapping
        train_gene_name_X_map = self.populate_X_y(mean_dict_train, X_train, y_train, embedding_size)
        test_gene_name_X_map = self.populate_X_y(mean_dict_test, X_test, y_test, embedding_size)

        # Convert lists to NumPy arrays
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_test, y_test = np.array(X_test), np.array(y_test)

        distance_results = calculate_distances(X_train, X_test)

        # Shape compatibility check
        if X_train.shape[1] != X_test.shape[1] or y_train.shape[1] != y_test.shape[1]:
            raise ValueError("Shape mismatch between training and testing sets.")

        run_results = {'ridge': {}, 'knn': {}}
        results_per_gene = {}

        # Ridge Regression evaluation
        for ridge_param in ridge_params:
            ridge_model = Ridge(**ridge_param)
            ridge_model.fit(X_train, y_train)
            y_pred_ridge = ridge_model.predict(X_test)

            # Evaluate row-wise (gene-wise) performance
            mse_ridge, mae_ridge, corr_ridge = self.evaluate_performance_rowwise(y_test, y_pred_ridge)
            run_results['ridge'][tuple(ridge_param.items())] = {'mse': np.mean(mse_ridge), 'mae': np.mean(mae_ridge), 'corr': np.mean(corr_ridge)}

            # Save per-gene performance results using `test_gene_name_X_map`
            for i, gene_name in enumerate(test_gene_name_X_map):
                if gene_name not in results_per_gene:
                    results_per_gene[gene_name] = {'ridge': {}, 'knn': {}}
                results_per_gene[gene_name]['ridge'][tuple(ridge_param.items())] = (
                    corr_ridge[i], mse_ridge[i], y_pred_ridge[i], y_test[i], 
                    distance_results['closest_distances'][i], distance_results['avg_top10_distances'][i]
                )

        # KNN evaluation
        for knn_param in knn_params:
            knn_model = KNeighborsRegressor(**knn_param)
            knn_model.fit(X_train, y_train)
            y_pred_knn = knn_model.predict(X_test)

            # Evaluate row-wise (gene-wise) performance
            mse_knn, mae_knn, corr_knn = self.evaluate_performance_rowwise(y_test, y_pred_knn)
            run_results['knn'][tuple(knn_param.items())] = {'mse': np.mean(mse_knn), 'mae': np.mean(mae_knn), 'corr': np.mean(corr_knn)}

            # Save per-gene performance results using `test_gene_name_X_map`
            for i, gene_name in enumerate(test_gene_name_X_map):
                if gene_name not in results_per_gene:
                    results_per_gene[gene_name] = {'ridge': {}, 'knn': {}}
                results_per_gene[gene_name]['knn'][tuple(knn_param.items())] = (
                    corr_knn[i], mse_knn[i], y_pred_knn[i], y_test[i],
                    distance_results['closest_distances'][i], distance_results['avg_top10_distances'][i]
                )

        # Return the final results
        results = {
            'aggregate': run_results,
            'per_gene': results_per_gene,
        }

        return results

    def run_kfold_experiments(self, ridge_params=None, knn_params=None, hidden_size=128, mlp_epochs=100, k=10, \
        use_mlp=False,condition_strategy_list = ['mean','median'],output_dir="./train_test_index",\
        mean_baseline=True):
        """
        Run the experiment using k-fold cross-validation and return average and std results across folds.

        Args:
        - ridge_params: list of dictionaries with hyperparameters for Ridge regression.
        - knn_params: list of dictionaries with hyperparameters for KNN regression.
        - hidden_size: hidden layer size for MLP (default: 128).
        - mlp_epochs: number of training epochs for MLP (default: 100).
        - k: number of folds for cross-validation (default: 10).

        Returns:
        - results_with_stats: A dictionary with average and std results across k folds for each model and per gene.
        """
        # Initialize storage for accumulating results
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        accumulated_results = {'ridge': {}, 'knn': {}, 'mlp': {}, 'per_gene': {}, 'train_condition': {}}

        # Get unique conditions to split by K-Fold
        unique_conditions = self.adata.obs['condition'].unique()

        # Use KFold cross-validation to create k splits
        kf = KFold(n_splits=k, shuffle=True, random_state=2024)

        for fold, (train_index, test_index) in enumerate(kf.split(unique_conditions)):
            print(f"Running fold {fold + 1}/{k}...")

            # Split into training and test conditions based on KFold indices
            train_conditions, test_conditions = unique_conditions[train_index], unique_conditions[test_index]

            # Run the experiment for the current train/test split
            results = self.run_experiment_with_conditions(
                train_conditions=train_conditions,
                test_conditions=test_conditions,
                ridge_params=ridge_params,
                knn_params=knn_params,
                hidden_size=hidden_size,
                mlp_epochs=mlp_epochs,
                use_mlp=use_mlp,
                condition_strategy_list=condition_strategy_list,
                mean_baseline=mean_baseline
            )

            # Save the train and test conditions to a JSON file
            # conditions_split = {
            #     'train': train_conditions.tolist(),
            #     'test': test_conditions.tolist()
            # }

            # json_filename = os.path.join(output_dir, f'fold_{fold + 1}_conditions.json')
            # with open(json_filename, 'w') as json_file:
            #     json.dump(conditions_split, json_file, indent=4)

            # Accumulate the results from this fold
            fold_results = results['aggregate']

            # Store the aggregate results for each model (ridge, knn, mlp)
            for model in ['ridge', 'knn', 'mlp', 'train_condition']:
                if model in fold_results:
                    for params, metrics in fold_results[model].items():
                        if params not in accumulated_results[model]:
                            accumulated_results[model][params] = {'mse': [], 'mae': [], 'corr': []}
                        accumulated_results[model][params]['mse'].append(metrics['mse'])
                        accumulated_results[model][params]['mae'].append(metrics['mae'])
                        accumulated_results[model][params]['corr'].append(metrics['corr'])
            # print('accumulated_results',accumulated_results)
            # Accumulate per-gene results for predictions and metrics, and include y_test
            for gene_name, model_results in results['per_gene'].items():
                # print("model_results['ridge']", model_results['ridge'])
                if gene_name not in accumulated_results['per_gene']:
                    accumulated_results['per_gene'][gene_name] = {'ridge': [], 'knn': [], 'mlp': []}
                if 'ridge' in model_results:
                    accumulated_results['per_gene'][gene_name]['ridge'] = model_results['ridge']
                if 'knn' in model_results:
                    accumulated_results['per_gene'][gene_name]['knn'] = model_results['knn']
                if 'mlp' in model_results:
                    accumulated_results['per_gene'][gene_name]['mlp'] = model_results['mlp']
                if 'train_condition' in model_results:
                    accumulated_results['per_gene'][gene_name]['train_condition'] = model_results['train_condition']

        # Calculate aggregate average and std for models
        results_with_stats = {'ridge': {}, 'knn': {}, 'mlp': {}, 'per_gene': {},'train_condition':{}}

        for model in ['ridge', 'knn', 'mlp','train_condition']:
            for params, metrics in accumulated_results[model].items():
                if metrics['mse']:  # Ensure there are values to calculate
                    results_with_stats[model][params] = {
                        'mean': {
                            'mse': np.mean(metrics['mse']),
                            'mae': np.mean(metrics['mae']),
                            'corr': np.mean(metrics['corr']),
                        },
                        'std': {
                            'mse': np.std(metrics['mse']),
                            'mae': np.std(metrics['mae']),
                            'corr': np.std(metrics['corr']),
                        }
                    }

        return results_with_stats, accumulated_results

