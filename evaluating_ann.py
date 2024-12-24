import time
import numpy as np
from typing import Dict, Callable, Tuple, List
import pandas as pd
from tqdm import tqdm
from IPython.display import clear_output


def evaluate_ann_search(
        embedded_images: List[np.ndarray],
        ann_methods: Dict[str, Callable],
        num_searches: int,
        pair_size: int = 2,
        K: int = 20,
        display=True
) -> pd.DataFrame:
    """
    Evaluates the performance of different ANN search functions on the provided embedded images.

    Args:
        embedded_images (List[np.ndarray]): List of embedded images, where each element is a
                                            numpy array representing the embeddings for one image.
        ann_methods (Dict[str, Callable]): Dictionary of ANN search functions, where the
                                           keys are the method names and the values are the
                                           search functions.
        num_searches (int): Number of searches to perform for each ANN method.
        pair_size (int, optional): Number of images per pair. Defaults to 2.
        K (int, optional): Number of nearest neighbors to retrieve. Defaults to 20.
        display (bool, optional): Whether to display additional information. Defaults to False.

    Returns:
        pd.DataFrame: Comprehensive performance metrics for each ANN search method
    """
    # Define comprehensive columns for performance metrics
    columns = [
        'Method Name',
        'Index Build Time',
        'Average Search Time',
        'Accuracy@K',
        'Recall@K',
        'Mean Average Precision@K',
        'Diversity Score'
    ]

    # Initialize results storage
    results = []

    # Ensure number of searches doesn't exceed available data
    num_searches = min(num_searches, len(embedded_images) // pair_size)

    # Randomly select search pairs
    pair_indices = np.random.choice(
        range(len(embedded_images) // pair_size),
        size=num_searches,
        replace=False
    ) * pair_size

    # Iterate through each ANN method
    for method_name, ANN_func in tqdm(ann_methods.items()):
        # print(f"Evaluating method: {method_name}")

        # Performance tracking variables
        search_times = []
        accuracies = []
        recalls = []
        map_scores = []
        diversity_scores = []

        # Index building time
        start_time = time.time()
        search_func = ANN_func(embedded_images)
        index_build_time = time.time() - start_time

        # Perform searches
        for index in pair_indices:
            # Run the search
            start_time = time.time()
            distances, indices = search_func(embedded_images[index], K)
            search_time = time.time() - start_time

            search_times.append(search_time)

            query_pairs = set([int(index) + i for i in range(1, pair_size)])
            set_indices = set(indices.tolist()[0][:K])
            matches = len(set_indices & query_pairs)

            # Calculate various metrics
            # Accuracy: Proportion of correct matches in top K results
            accuracies.append(matches / K)

            # Recall: Proportion of ground truth neighbors found
            recalls.append(matches / (pair_size - 1))

            # Mean Average Precision
            precisions = []
            for i in range(1, K + 1):
                precision_at_k = len(set(indices.tolist()[0][:i]) & query_pairs) / i
                precisions.append(precision_at_k)
            map_scores.append(np.mean(precisions))

            # Diversity Score: Measures the variety of retrieved results
            # Calculate variance of distances as a simple diversity metric
            diversity_scores.append(np.std(distances))

        # Compile method-level metrics
        method_results = {
            'Method Name': method_name,
            'Mean Average Precision@K': np.mean(map_scores),
            'Index Build Time': index_build_time,
            'Average Search Time': np.mean(search_times),
            'Accuracy@K': np.mean(accuracies),
            'Recall@K': np.mean(recalls),
            # 'Diversity Score': np.mean(diversity_scores)
        }

        results.append(method_results)

    # clear_output(wait=False) # it messes with the future cells!?
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Optional display
    if display:
        print(results_df)

    return results_df