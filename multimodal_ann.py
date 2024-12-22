import numpy as np
import faiss
from typing import List, Callable, Tuple, Set, Dict
from dataclasses import dataclass

# Type aliases for clarity
EmbeddingsList = List[List[np.ndarray]]  # List of lists of embeddings
SearchFunction = Callable[[List[np.ndarray]], Tuple[np.ndarray, np.ndarray]]


@dataclass
class RobustSearchConfig:
    min_embeddings_required: int = 1  # Minimum embeddings needed for a match
    use_rank_fusion: bool = True  # Whether to use rank-based fusion
    outlier_threshold: float = 2.0  # Z-score threshold for outlier detection
    rank_weights: List[float] = None  # Weights for different embedding types


class MultimodalANN:
    def __init__(self, init=True):
        self.methods_list = {}

        if init:
            self.methods_list["hnsw_simple_concatenation"] = self.hnsw_simple_concatenation_ann
            self.methods_list["LSH_simple_concatenation"] = self.lsh_simple_concatenation_ann
            self.methods_list["separate indexing"] = self.multi_index_ann
            self.methods_list["Split LSH"] = self.split_lsh_ann
            self.methods_list["just norm"] = self.just_norm_ann
            self.methods_list["Normalized and Scaled"] = self.normalized_scaled_ann
            # self.methods_list["learn_class_weights"] = self.wrapper_learn_class_weights  # TODO: check this func
            self.methods_list["Dimension Reduction"] = self.dimension_reduction_ann
            self.methods_list["robust multi index"] = self.robust_multi_index_ann
            self.methods_list["Capped Distance"] = self.capped_distance_ann
            self.methods_list["Tolerant ANN"] = self.tolerant_ann
            self.methods_list["Emphasis Close"] = self.emphasis_close_ann

    # -------------------------------------------- ANN FUNCTIONS -----------------------------------------------------
    # These functions are in self-contained callable form, meaning the indexing function returns the search function

    # Concatenation of Embeddings (HNSW and LSH) ---------------------------------------------------------------------
    # The simplest fusion strategy involves concatenating all embedding vectors from the different modalities.
    # This approach treats each embedding as a separate dimension in a single, unified vector space without any further
    # processing or weighting. This strategy was tested on HNSW and LSH.
    @staticmethod
    def hnsw_simple_concatenation_ann(data: EmbeddingsList) -> SearchFunction:
        """
        Simple concatenation method. Concatenates embeddings as-is.

        Args:
            data: List where each entry is a list of embeddings (one per embedding type)

        Returns:
            search_function: Function that takes query embeddings and returns distances and indices
        """
        # Validate and process data
        if not data or not all(data):
            raise ValueError("Data must be non-empty and all entries must have embeddings")

        # Get dimensions and validate consistency
        num_embedding_types = len(data[0])
        dims = [data[0][i].shape[-1] for i in range(num_embedding_types)]
        total_dim = sum(dims)

        # Concatenate all data
        concatenated_data = np.hstack([
            np.vstack([entry[i] for entry in data])
            for i in range(num_embedding_types)
        ]).astype('float32')

        # Create and train index
        index = faiss.IndexHNSWFlat(total_dim, 16)
        index.hnsw.efConstruction = 100
        index.hnsw.efSearch = 64
        index.add(concatenated_data)

        def search(query_embeddings: List[np.ndarray], k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
            if len(query_embeddings) != num_embedding_types:
                raise ValueError(f"Expected {num_embedding_types} embedding types, got {len(query_embeddings)}")

            concatenated_query = np.hstack(query_embeddings).astype('float32').reshape((1, -1))
            distances, indices = index.search(concatenated_query, k)
            return distances, indices

        return search

    @staticmethod
    def lsh_simple_concatenation_ann(data: EmbeddingsList, num_planes: int = 256) -> SearchFunction:
        """
        Normal LSH method. Applies LSH to concatenated embeddings.

        Args:
            data: List where each entry is a list of embeddings (one per embedding type)
            num_planes: Number of hyperplanes for LSH

        Returns:
            search_function: Function that takes query embeddings and returns distances and indices
        """
        if not data or not all(data):
            raise ValueError("Data must be non-empty and all entries must have embeddings")

        num_embedding_types = len(data[0])
        dims = [data[0][i].shape[-1] for i in range(num_embedding_types)]
        total_dim = sum(dims)

        # Concatenate all data
        concatenated_data = np.hstack([
            np.vstack([entry[i] for entry in data])
            for i in range(num_embedding_types)
        ]).astype('float32')

        # Create and train LSH index
        index = faiss.IndexLSH(total_dim, num_planes)
        index.train(concatenated_data)
        index.add(concatenated_data)

        def search(query_embeddings: List[np.ndarray], k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
            if len(query_embeddings) != num_embedding_types:
                raise ValueError(f"Expected {num_embedding_types} embedding types, got {len(query_embeddings)}")

            concatenated_query = np.hstack(query_embeddings).astype('float32').reshape(1, -1)
            distances, indices = index.search(concatenated_query, k)
            return distances, indices

        return search

    # Separate Indexing ----------------------------------------------------------------------------------------------
    # In this method, each embedding type is indexed separately.
    # During the search phase, each embedding is queried independently, and the results are combined based on the
    # respective similarity scores. This approach allows each embedding type to maintain its own search space but
    # does not account for the interactions between the different types.
    @staticmethod
    def multi_index_ann(data: EmbeddingsList) -> SearchFunction:
        """
        Creates separate indices for each embedding type and combines their results,
        calculating distances to all embeddings for each candidate.

        Args:
            data: List where each entry is a list of embeddings (one per embedding type)

        Returns:
            search_function: Function that takes query embeddings and returns combined results
        """
        if not data or not all(data):
            raise ValueError("Data must be non-empty and all entries must have embeddings")

        num_embedding_types = len(data[0])
        entry_count = len(data)
        dims = [data[0][i].shape[-1] for i in range(num_embedding_types)]

        # Create separate indices and store embeddings for each type
        indices = []
        embeddings_by_type = []

        for i in range(num_embedding_types):
            embeddings = np.vstack([entry[i] for entry in data]).astype('float32')
            embeddings_by_type.append(embeddings)

            index = faiss.IndexHNSWFlat(dims[i], 16)
            index.hnsw.efConstruction = 100
            index.hnsw.efSearch = 64
            index.add(embeddings)
            indices.append(index)

        def search(query_embeddings: List[np.ndarray], k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
            if len(query_embeddings) != num_embedding_types:
                raise ValueError(f"Expected {num_embedding_types} embedding types, got {len(query_embeddings)}")

            # Search in each index separately to get initial candidate set
            all_candidates: Set[int] = set()

            for i, (query, index) in enumerate(zip(query_embeddings, indices)):
                query_array = query.reshape(1, -1).astype('float32')
                _, indices_found = index.search(query_array, k)
                all_candidates.update(indices_found[0])

            # Calculate distances for all candidates across all embedding types
            candidate_scores = []

            for idx in all_candidates:
                total_distance = 0
                for i, query in enumerate(query_embeddings):
                    # Reshape for broadcasting
                    query_vector = query.reshape(1, -1)
                    db_vector = embeddings_by_type[i][idx].reshape(1, -1)

                    # Calculate L2 distance
                    distance = np.linalg.norm(query_vector - db_vector)
                    total_distance += distance

                candidate_scores.append((idx, total_distance))

            # Sort by total distance and get top k
            candidate_scores.sort(key=lambda x: x[1])
            final_indices = [idx for idx, _ in candidate_scores[:k]]
            final_distances = [dist for _, dist in candidate_scores[:k]]

            return (np.array(final_distances).reshape(1, -1),
                    np.array(final_indices).reshape(1, -1))

        return search

    @staticmethod
    def split_lsh_ann(data: EmbeddingsList, num_planes: int = 256) -> SearchFunction:
        """
        Split LSH method. Applies LSH separately to each embedding type.

        Args:
            data: List where each entry is a list of embeddings (one per embedding type)
            num_planes: Total number of hyperplanes to be distributed among embedding types

        Returns:
            search_function: Function that takes query embeddings and returns distances and indices
        """
        if not data or not all(data):
            raise ValueError("Data must be non-empty and all entries must have embeddings")

        num_embedding_types = len(data[0])
        dims = [data[0][i].shape[-1] for i in range(num_embedding_types)]
        total_dim = sum(dims)

        # Allocate planes proportionally to dimension sizes
        planes_per_type = [max(1, int(num_planes * dim / total_dim)) for dim in dims]

        # Adjust to ensure total equals num_planes
        while sum(planes_per_type) != num_planes:
            if sum(planes_per_type) < num_planes:
                idx = dims.index(max(dims))
                planes_per_type[idx] += 1
            else:
                idx = dims.index(min(dims))
                if planes_per_type[idx] > 1:
                    planes_per_type[idx] -= 1

        # Create separate LSH indexes for each embedding type
        indexes = []
        for i, dim in enumerate(dims):
            index = faiss.IndexLSH(total_dim, planes_per_type[i])

            # Create masked training data
            masked_data = np.zeros((len(data), total_dim), dtype='float32')
            start_idx = sum(dims[:i])
            masked_data[:, start_idx:start_idx + dims[i]] = np.vstack([entry[i] for entry in data])

            index.train(masked_data)
            index.add(masked_data)
            indexes.append(index)

        def search(query_embeddings: List[np.ndarray], k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
            if len(query_embeddings) != num_embedding_types:
                raise ValueError(f"Expected {num_embedding_types} embedding types, got {len(query_embeddings)}")

            concatenated_query = np.hstack(query_embeddings).astype('float32').reshape(1, -1)

            # Combine results from all indexes
            all_distances = []
            all_indices = []

            for i, index in enumerate(indexes):
                distances, indices = index.search(concatenated_query, k)
                all_distances.append(distances)
                all_indices.append(indices)

            # Merge and sort results
            merged_distances = np.hstack(all_distances)
            merged_indices = np.hstack(all_indices)

            # Sort by distance and take top k
            sorted_indices = np.argsort(merged_distances[0])[:k]
            final_distances = merged_distances[0][sorted_indices].reshape(1, -1)
            final_indices = merged_indices[0][sorted_indices].reshape(1, -1)

            return final_distances, final_indices

        return search

    # Normalized and Scaled Embeddings ---------------------------------------------------------------------------
    # To improve the performance of the concatenation method, embeddings are first normalized (to unit length)
    # and scaled. This ensures that embeddings with different ranges and magnitudes are brought to a common scale,
    # making them more comparable in terms of distance calculations during the search phase.
    @staticmethod
    def just_norm_ann(data: EmbeddingsList) -> SearchFunction:
        """
        Standard ANN: normalize each embedding to have a cosine-similarity-like search score.
        """
        num_embedding_types = len(data[0])
        dims = [data[0][i].shape[-1] for i in range(num_embedding_types)]

        # Normalize embeddings for consistent distance scaling
        normalized_data = []
        for i in range(num_embedding_types):
            embeddings = np.vstack([entry[i] for entry in data])
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 0.001
            normalized_data.append(embeddings / norms)

        # Create separate indexes for each embedding type for efficient search
        indexes = []
        for i, emb in enumerate(normalized_data):
            index = faiss.IndexHNSWFlat(dims[i], 16)
            index.hnsw.efConstruction = 100
            index.hnsw.efSearch = 64
            index.add(emb.astype('float32'))
            indexes.append(index)

        def search(query_embeddings: List[np.ndarray], k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
            if len(query_embeddings) != num_embedding_types:
                raise ValueError(f"Expected {num_embedding_types} embedding types, got {len(query_embeddings)}")

            # Normalize query embeddings
            normalized_queries = []
            for query_emb in query_embeddings:
                norm = np.linalg.norm(query_emb) + 0.001
                normalized_queries.append((query_emb / norm).astype('float32').reshape(1, -1))

            # Initial wide search in each embedding space
            expanded_k = min(k * 2, len(data))  # Search for more candidates initially
            all_candidates = set()
            per_type_results = []

            for i, (index, query) in enumerate(zip(indexes, normalized_queries)):
                distances, indices = index.search(query, expanded_k)
                all_candidates.update(indices[0])
                per_type_results.append((distances[0], indices[0]))

            # Calculate final scores for all candidates
            final_scores = []
            for idx in all_candidates:
                type_distances = []
                # Collect distances for this candidate across all embedding types
                for type_idx, (distances, indices) in enumerate(per_type_results):
                    if idx in indices:
                        dist = distances[np.where(indices == idx)[0][0]]
                        type_distances.append(dist)
                    else:
                        # If this candidate wasn't in top-k for this embedding type,
                        # use a default high distance
                        type_distances.append(1.0)

                # Calculate final score
                final_score = sum(type_distances) / len(type_distances)
                final_scores.append((final_score, idx))

            # Sort and get top k
            final_scores.sort()
            top_k = final_scores[:k]

            # Format results
            result_distances = np.array([[score for score, _ in top_k]])
            result_indices = np.array([[idx for _, idx in top_k]])

            return result_distances, result_indices

        return search

    @staticmethod
    def normalized_scaled_ann(data: EmbeddingsList) -> SearchFunction:
        """
        Normalized and scaled concatenation method. Normalizes each embedding type
        and scales based on dimension size.

        Args:
            data: List where each entry is a list of embeddings (one per embedding type)

        Returns:
            search_function: Function that takes query embeddings and returns distances and indices
        """
        if not data or not all(data):
            raise ValueError("Data must be non-empty and all entries must have embeddings")

        num_embedding_types = len(data[0])
        dims = [data[0][i].shape[-1] for i in range(num_embedding_types)]
        total_dim = sum(dims)

        # Calculate scaling factors based on dimensions
        max_dim = max(dims)
        scaling_factors = [np.sqrt(max_dim / dim) for dim in dims]

        # Normalize and scale each embedding type, then concatenate
        # do Z-score normalization.
        # Find the maximum length of the embeddings in each column
        numberOfEmbeddings = len(data[0])
        max_lengths = [len(data[0][i]) for i in range(numberOfEmbeddings)]

        # Initialize lists to store sums and counts for each column
        sums = [np.zeros(mlengths) for mlengths in max_lengths]
        counts = len(data)

        # First pass: Calculate sums and counts for each index of embeddings in each column
        for row in data:
            for i in range(numberOfEmbeddings):
                sums[i] += row[i]

        # Calculate means for each index of embeddings in each column
        means = [su / counts for su in sums]

        # Initialize lists to store squared differences for each column
        squared_diffs = [np.zeros(mlengths) for mlengths in max_lengths]

        # Second pass: Calculate squared differences from the mean for each index of embeddings in each column
        for row in data:
            for i in range(numberOfEmbeddings):
                squared_diffs[i] += np.power(row[i] - means[i], 2)

        # Calculate standard deviations for each index of embeddings in each column
        stds = [np.sqrt(squared_diff / counts) for squared_diff in squared_diffs]

        # to avoid division by zero
        for std in stds:
            std += (std == 0)  # mask is 1 where std == 0
        # Normalize the data
        normalized_data = []
        for row in data:
            normalized_row = []
            for mean, std, ro in zip(means, stds, row):
                normalized_row.append((ro - mean) / std)
            normalized_data.append(normalized_row)

        concatenated_data = np.hstack([
            np.vstack([entry[i] for entry in normalized_data]) * scaling_factors[i]
            for i in range(num_embedding_types)
        ]).astype('float32')

        # Create and train index
        index = faiss.IndexHNSWFlat(total_dim, 16)
        index.hnsw.efConstruction = 100
        index.hnsw.efSearch = 64
        index.add(concatenated_data)

        def search(query_embeddings: List[np.ndarray], k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
            if len(query_embeddings) != num_embedding_types:
                raise ValueError(f"Expected {num_embedding_types} embedding types, got {len(query_embeddings)}")

            # Apply the same normalization and scaling to query embeddings
            normalized_scaled_query = np.hstack([
                ((query_embeddings[i] - means[i]) / stds[i]) * scaling_factors[i]
                for i in range(num_embedding_types)
            ]).astype('float32').reshape(1, -1)

            distances, indices = index.search(normalized_scaled_query, k)
            # todo efficiently find real distances before normalizing
            distances = []
            unnormalized_query = np.hstack([i for i in query_embeddings])
            for idx in indices[0]:
                unnormalized_vector = np.hstack([i for i in data[idx]])
                distances.append(np.linalg.norm(unnormalized_vector - unnormalized_query))
            distances = np.array([distances]).reshape((1, -1))
            return distances, indices

        return search

    # Learning weights -----------------------------------------------------------------------------------------------
    # Weighted fusion is a common baseline fusion strategy when the weights are known beforehand.
    # In this approach, we use the indexing data and their class index to very quickly learn helpful weights by
    # comparing cluster class size under each embedding type and other quick indicators to identify stronger embeddings.
    @staticmethod
    def wrapper_learn_class_weights(class_ids: List[int], stop_in=20) -> List[float]:
        def indexing_stage(data: EmbeddingsList) -> SearchFunction:
            """
            Learn the weights for each embedding type based on distances to the class center.
            The goal is to minimize the intra-class distance while maximizing the inter-class distance using
            learned weights.

            Args:
                data: List where each entry is a list of embeddings (one per embedding type).
                class_ids: List of class IDs corresponding to each entry in `data`.

            Returns:
                List of learned weights for each embedding type.
            """
            if not data or not class_ids:
                raise ValueError("Data and class_ids must not be empty.")
            if len(data) != len(class_ids):
                raise ValueError("Data and class_ids must have the same length.")

            num_samples = len(class_ids)
            num_embedding_types = len(data[0])

            # Initialize weights for each embedding type (starting with equal weights)
            weights = np.ones(num_embedding_types)

            # Compute class centers (mean) for each embedding type
            class_centers = {}
            # Store intra-class and extra-class distances for each embedding_type
            intra_class_distances = {}
            extra_class_distances = []
            extra_closer_intra = [0 for _ in range(num_embedding_types)]

            stop_in_i = stop_in
            for class_id in np.unique(class_ids):
                stop_in_i -= 1
                if stop_in_i < 0:
                    break
                # Get the embeddings for each class
                class_indices = np.where(class_ids == class_id)[0]
                class_embeddings = [np.vstack([data[i][k] for i in class_indices]) for k in range(num_embedding_types)]

                # Calculate the center (mean) of each embedding type for this class
                class_centers[class_id] = [np.mean(embedding, axis=0) for embedding in class_embeddings]

                # Intra-class distances: distance to the center of the same class
                intra_dist = []
                for embedding, center in zip(class_embeddings, class_centers[class_id]):
                    dist = np.linalg.norm(embedding - center, axis=1)
                    intra_dist.append(np.mean(dist))
                intra_class_distances[class_id.item()] = intra_dist

            # Loop through each class centers to sample and calculate distances
            for class_id, class_center in class_centers.items():
                not_class_indices = np.where(class_ids != class_id)[0]
                not_class_embeddings = [np.vstack([data[i][k] for i in not_class_indices]) for k in
                                        range(num_embedding_types)]
                extra_dist = []
                for i, (embedding, center) in enumerate(zip(not_class_embeddings, class_center)):
                    dist = np.linalg.norm(embedding - center, axis=1)
                    extra_closer_intra[i] += np.count_nonzero(dist < intra_class_distances[class_id][i])
                    extra_dist.append(np.mean(dist))
                extra_class_distances.append(extra_dist)

            # Calculate ratio (intra-class / inter-class distance)
            intra_class_distances_mean = [0 for _ in range(num_embedding_types)]
            for value in intra_class_distances.values():
                for i, v in enumerate(value):
                    intra_class_distances_mean[i] += v
            for i in range(len(intra_class_distances_mean)):
                intra_class_distances_mean[i] /= len(intra_class_distances)
            extra_class_distances_mean = [0 for _ in range(num_embedding_types)]
            for value in extra_class_distances:
                for i, v in enumerate(value):
                    extra_class_distances_mean[i] += v
            for i in range(len(extra_class_distances_mean)):
                extra_class_distances_mean[i] /= len(extra_class_distances)
            distance_ratios = [ext / intra for ext, intra in
                               zip(extra_class_distances_mean, intra_class_distances_mean)]

            # Use this ratio to adjust weights
            for i in range(num_embedding_types):
                # Weight will be proportional to the ratio of extra-class to intra-class distance
                if distance_ratios[i] < 1:
                    weights[i] = 0
                else:  # add some weight if few close to center
                    weights[i] = np.power(distance_ratios[i], 2) + -extra_closer_intra[i] / len(
                        data)  # We can tweak this scaling
            weights /= np.mean(weights)
            weights = weights.tolist()

            print("during the training process of learn_class , found the weights for each embedding are", weights)

            dims = [data[0][i].shape[-1] for i in range(num_embedding_types)]

            # Create separate indices and store embeddings for each type

            embeddings_by_type = []
            embeddings = np.hstack([np.vstack([entry[i] * w for entry in data]) for i, w in enumerate(weights)])

            index = faiss.IndexHNSWFlat(sum(dims), 16)
            index.hnsw.efConstruction = 100
            index.hnsw.efSearch = 64
            index.add(embeddings)

            def search(query_embeddings: List[np.ndarray], k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
                if len(query_embeddings) != num_embedding_types:
                    raise ValueError(f"Expected {num_embedding_types} embedding types, got {len(query_embeddings)}")

                # Calculate weighted average of query embeddings using learned weights
                weighted_query = np.hstack([query_embeddings[i] * w for i, w in enumerate(weights)]).reshape((1, -1))
                # print(index.search(weighted_query, k))
                distances, indices = index.search(weighted_query, k)
                return distances, indices

            return search

        return indexing_stage  # TODO: why the wrapper?

    # def raise_error(x):
    #     raise "wrapper not started with class_ids"
    #
    # methods_list["learn_class_weights"] = raise_error TODO: what is that?

    # Dimension Reduction (PCA) --------------------------------------------------------------------------------------
    # Principal Component Analysis (PCA) is applied to reduce the dimensionality of the combined embeddings.
    # The goal is to extract the most important features while discarding noise and redundant information.
    # This reduces the computational cost of the search while maintaining as much discriminative power as possible.
    @staticmethod
    def dimension_reduction_ann(data: EmbeddingsList, target_dim: int = 128) -> SearchFunction:
        """
        Dimension reduction method. Reduces total dimensions to a target size.

        Args:
            data: List where each entry is a list of embeddings (one per embedding type)
            target_dim: Target dimension for the reduced space (default 128)

        Returns:
            search_function: Function that takes query embeddings and returns distances and indices
        """
        # Validate and process data
        if not data or not all(data):
            raise ValueError("Data must be non-empty and all entries must have embeddings")

        # Get dimensions and validate consistency
        num_embedding_types = len(data[0])
        dims = [data[0][i].shape[-1] for i in range(num_embedding_types)]
        total_original_dim = sum(dims)

        target_dim = min(target_dim, round(total_original_dim * 3 / 4))

        # Concatenate all data
        concatenated_data = np.hstack([
            np.vstack([entry[i] for entry in data])
            for i in range(num_embedding_types)
        ]).astype('float32')

        # Create PCA reducer
        reducer = faiss.PCAMatrix(total_original_dim, target_dim)

        # Train reducer on the data
        reducer.train(concatenated_data)

        # Apply reduction to the data
        reduced_data = reducer.apply_py(concatenated_data)

        # Create and train index on reduced data
        index = faiss.IndexHNSWFlat(target_dim, 16)
        index.hnsw.efConstruction = 100
        index.hnsw.efSearch = 64
        index.add(reduced_data)

        def search(query_embeddings: List[np.ndarray], k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
            if len(query_embeddings) != num_embedding_types:
                raise ValueError(f"Expected {num_embedding_types} embedding types, got {len(query_embeddings)}")

            # Concatenate query embeddings
            concatenated_query = np.hstack(query_embeddings).astype('float32').reshape(1, -1)

            # Reduce query dimensions
            reduced_query = reducer.apply_py(concatenated_query)

            # Perform search
            distances, indices = index.search(reduced_query, k)
            return distances, indices

        return search

    # Robust Multi-Indexing ------------------------------------------------------------------------------------------
    # This strategy normalizes and weights each embedding type individually and performs searches within each
    # modality's respective index. The results from each search are then merged using reciprocal rank, which adjusts
    # the final ranking based on the relative importance of each embedding type. This approach accounts for the
    # varying relevance of each modality in the final result.
    @staticmethod
    def robust_multi_index_ann(data: EmbeddingsList) -> SearchFunction:
        """
        Creates a robust ANN search that can handle unreliable embeddings.

        Args:
            data: List where each entry is a list of embeddings (one per embedding type)

        Returns:
            search_function: Function that takes query embeddings and returns robust results
        """
        if not data or not all(data):
            raise ValueError("Data must be non-empty and all entries must have embeddings")

        num_embedding_types = len(data[0])
        entry_count = len(data)
        dims = [data[0][i].shape[-1] for i in range(num_embedding_types)]

        # Create indices and store embeddings
        indices = []
        embeddings_by_type = []

        # Calculate mean and std for each embedding type for normalization
        means = []
        stds = []

        for i in range(num_embedding_types):
            embeddings = np.vstack([entry[i] for entry in data]).astype('float32')
            embeddings_by_type.append(embeddings)

            # Store mean and std for normalization
            means.append(np.mean(embeddings, axis=0))
            stds.append(np.std(embeddings, axis=0))

            index = faiss.IndexHNSWFlat(dims[i], 16)
            index.hnsw.efConstruction = 100
            index.hnsw.efSearch = 64
            index.add(embeddings)
            indices.append(index)

        def reciprocal_rank_fusion(rankings: List[List[int]], k: int = 60) -> Dict[int, float]:
            """Combine multiple rankings using reciprocal rank fusion"""
            fused_scores = {}
            for rank_list in rankings:
                for rank, idx in enumerate(rank_list):
                    if idx not in fused_scores:
                        fused_scores[idx] = 0
                    fused_scores[idx] += 1.0 / (rank + k)
            return fused_scores

        def calculate_normalized_distances(query: np.ndarray, embeddings: np.ndarray,
                                           mean: np.ndarray, std: np.ndarray) -> np.ndarray:
            """Calculate normalized distances accounting for distribution"""
            normalized_query = (query - mean) / std
            normalized_embeddings = (embeddings - mean) / std
            return np.linalg.norm(normalized_embeddings - normalized_query, axis=1)

        def search(query_embeddings: List[np.ndarray], k: int = 5,
                   config: RobustSearchConfig = RobustSearchConfig()) -> Tuple[np.ndarray, np.ndarray]:
            if len(query_embeddings) != num_embedding_types:
                raise ValueError(f"Expected {num_embedding_types} embedding types, got {len(query_embeddings)}")

            # Initialize storage for rankings and distances
            all_rankings: List[List[int]] = []
            all_distances: List[Dict[int, float]] = []
            all_candidates: Set[int] = set()

            # Search in each index and store results
            for i, (query, index) in enumerate(zip(query_embeddings, indices)):
                query_array = query.reshape(1, -1).astype('float32')
                distances, indices_found = index.search(query_array, k)

                # Store rankings and distances
                all_rankings.append(indices_found[0].tolist())
                all_candidates.update(indices_found[0])

                # Calculate normalized distances for all candidates
                normalized_distances = calculate_normalized_distances(
                    query, embeddings_by_type[i], means[i], stds[i])
                all_distances.append(
                    {idx: dist for idx, dist in enumerate(normalized_distances)})  # TODO: check and fix

            # Combine results using various robust methods
            candidate_scores = []

            if config.use_rank_fusion:
                # Use rank fusion for initial scoring
                fused_scores = reciprocal_rank_fusion(all_rankings)
                initial_candidates = set(fused_scores.keys())
            else:
                initial_candidates = all_candidates

            weights = config.rank_weights or [1.0] * num_embedding_types

            for idx in initial_candidates:
                embedding_scores = []
                valid_embeddings = 0

                for i in range(num_embedding_types):
                    distance = all_distances[i][idx]

                    # Check if this embedding is an outlier
                    if abs(distance - np.mean(list(all_distances[i].values()))) < \
                            config.outlier_threshold * np.std(list(all_distances[i].values())):
                        embedding_scores.append(distance * weights[i])
                        valid_embeddings += 1

                # Only consider if we have enough valid embeddings
                if valid_embeddings >= config.min_embeddings_required:
                    # Use median of scores to be robust to outliers
                    median_score = np.median(embedding_scores)
                    candidate_scores.append((idx, median_score))

            # Sort and get top k
            candidate_scores.sort(key=lambda x: x[1])
            final_indices = [idx for idx, _ in candidate_scores[:k]]
            final_distances = [dist for _, dist in candidate_scores[:k]]

            return (np.array(final_distances).reshape(1, -1),
                    np.array(final_indices).reshape(1, -1))

        return search

    # Capped Distance ------------------------------------------------------------------------------------------------
    # In this approach, the maximum allowable distance between any two embeddings is capped for each embedding type.
    # This limits the search space and helps to reduce the influence of outlier embeddings, ensuring that only the most
    # relevant candidates are considered in the final search.
    @staticmethod
    def capped_distance_ann(data: EmbeddingsList, cap_distance: float = 0.55) -> SearchFunction:
        """
        ANN method using capped distances for each embedding type.
        Limits the impact of outlier embeddings by capping their contribution to the total distance.

        Args:
            data: List where each entry is a list of embeddings (one per embedding type)
            cap_distance: Maximum distance contribution from each embedding type

        Returns:
            search_function: Function that takes query embeddings and returns distances and indices
        """
        if not data or not all(data):
            raise ValueError("Data must be non-empty and all entries must have embeddings")

        num_embedding_types = len(data[0])
        dims = [data[0][i].shape[-1] for i in range(num_embedding_types)]
        total_dim = sum(dims)

        # Concatenate all data
        concatenated_data = np.hstack([
            np.vstack([entry[i] for entry in data])
            for i in range(num_embedding_types)
        ]).astype('float32')

        class CappedDistanceIndex:
            def __init__(self, data, dims, cap):
                self.data = data
                self.dims = dims
                self.cap = cap
                self.dim_starts = [0] + list(np.cumsum(dims[:-1]))

            def search(self, query, k):
                # Compute distances for each embedding type separately
                n = len(self.data)
                all_distances = np.zeros((len(query), n), dtype=np.float32)

                for i, start in enumerate(self.dim_starts):
                    end = start + self.dims[i]
                    # Compute squared L2 distance for this embedding type
                    delta = self.data[:, start:end] - query[:, start:end]
                    distances = np.sum(delta * delta, axis=1)
                    # Cap the distances
                    np.minimum(distances, self.cap, out=distances)

                    all_distances += distances

                # Use argpartition to efficiently find top k
                ind = np.argpartition(all_distances[0], k)[:k]
                # Sort the top k
                ind_sorted = ind[np.argsort(all_distances[0][ind])]
                distances_sorted = all_distances[0][ind_sorted]

                return np.sqrt(distances_sorted.reshape(1, -1)), ind_sorted.reshape(1, -1)

        # Create the index
        index = CappedDistanceIndex(concatenated_data, dims,
                                    0.3)  # cap_distance) #this function is cheating and not ANN, how created it

        def search(query_embeddings: List[np.ndarray], k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
            if len(query_embeddings) != num_embedding_types:
                raise ValueError(f"Expected {num_embedding_types} embedding types, got {len(query_embeddings)}")

            concatenated_query = np.hstack(query_embeddings).astype('float32').reshape(1, -1)
            distances, indices = index.search(concatenated_query, k)
            return distances, indices

        return search

    # Tolerant Search:
    # This method ranks the results for each embedding type separately and then smooths the distances between embeddings
    # by adjusting the proximity based on the number of close matches from each embedding.
    # This helps to mitigate the impact of noisy or mismatched embeddings by promoting a more stable and consistent ranking.
    @staticmethod
    def tolerant_ann(data: EmbeddingsList, subset_size: int = None) -> SearchFunction:
        """
        Outlier-tolerant ANN method. Creates multiple indexes using different subsets of embeddings.
        Finds matches that are close in most, but not necessarily all, embedding spaces.

        Args:
            data: List where each entry is a list of embeddings (one per embedding type)
            subset_size: Number of embedding types to combine in each subset (default: num_types - 1)

        Returns:
            search_function: Function that takes query embeddings and returns distances and indices
        """
        if not data or not all(data):
            raise ValueError("Data must be non-empty and all entries must have embeddings")

        num_embedding_types = len(data[0])
        dims = [data[0][i].shape[-1] for i in range(num_embedding_types)]

        if subset_size is None:
            subset_size = max(1, num_embedding_types - 1)

        if subset_size > num_embedding_types:
            raise ValueError(
                f"subset_size ({subset_size}) cannot be larger than number of embedding types ({num_embedding_types})")

        # Create all possible combinations of embedding types
        from itertools import combinations
        embedding_combinations = list(combinations(range(num_embedding_types), subset_size))

        # Create an index for each combination
        indexes = []
        for combo in embedding_combinations:
            # Calculate total dimension for this combination
            combo_dim = sum(dims[i] for i in combo)

            # Concatenate only the embeddings in this combination
            combo_data = np.hstack([
                np.vstack([entry[i] for entry in data])
                for i in combo
            ]).astype('float32')

            # Create and train index
            index = faiss.IndexHNSWFlat(combo_dim, 16)
            index.hnsw.efConstruction = 100
            index.hnsw.efSearch = 64
            index.add(combo_data)

            indexes.append((combo, index))

        def search(query_embeddings: List[np.ndarray], k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
            if len(query_embeddings) != num_embedding_types:
                raise ValueError(f"Expected {num_embedding_types} embedding types, got {len(query_embeddings)}")

            # Search in each index and aggregate results
            all_results = {}  # Dictionary to store index -> score mapping

            for combo, index in indexes:
                # Concatenate only the query embeddings for this combination
                combo_query = np.hstack([query_embeddings[i] for i in combo]).astype('float32').reshape(1, -1)

                # Search in this index
                distances, indices = index.search(combo_query, k)

                # Accumulate scores for each found index
                for idx, dist in zip(indices[0], distances[0]):
                    if idx not in all_results:
                        all_results[idx] = []
                    all_results[idx].append(dist)

            # Aggregate scores using a method that rewards consistency across subsets
            final_scores = []
            for idx, distances in all_results.items():
                # Use the average of best scores as the final score
                best_scores = sorted(distances)[:max(1, len(indexes) - num_embedding_types + subset_size)]
                score = sum(best_scores) / len(best_scores)
                final_scores.append((score, idx))

            # Sort by score and get top k
            final_scores.sort()
            top_k = final_scores[:k]

            # Format results to match expected output
            result_distances = np.array([[score for score, _ in top_k]])
            result_indices = np.array([[idx for _, idx in top_k]])

            return result_distances, result_indices

        return search

    # Emphasis Closeness ---------------------------------------------------------------------------------------------
    # The emphasis closeness method adjusts the distance between embeddings by lowering the cost of other embedding
    # types if some embeddings are very close to each other. This approach emphasizes the strength of close matches
    # in one modality and allows the other modalities to contribute more strongly if they align closely,
    # improving overall search accuracy.
    @staticmethod
    def emphasis_close_ann(data: EmbeddingsList, boost_threshold: float = 0.3,
                           default_high_distance=0.4) -> SearchFunction:
        """
        ANN method that emphasizes close matches in any embedding type.
        If any embedding type shows a very close match, it reduces the impact of other distances.

        Args:
            data: List where each entry is a list of embeddings (one per embedding type)
            boost_threshold: Distance threshold below which a match is considered "close"
                            and triggers emphasis behavior

        Returns:
            search_function: Function that takes query embeddings and returns distances and indices
        """
        if not data or not all(data):
            raise ValueError("Data must be non-empty and all entries must have embeddings")

        num_embedding_types = len(data[0])
        dims = [data[0][i].shape[-1] for i in range(num_embedding_types)]

        # Normalize embeddings for consistent distance scaling
        normalized_data = []
        for i in range(num_embedding_types):
            embeddings = np.vstack([entry[i] for entry in data])
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 0.001
            normalized_data.append(embeddings / norms)

        # Create separate indexes for each embedding type for efficient search
        indexes = []
        for i, emb in enumerate(normalized_data):
            index = faiss.IndexHNSWFlat(dims[i], 16)
            index.hnsw.efConstruction = 100
            index.hnsw.efSearch = 64
            index.add(emb.astype('float32'))
            indexes.append(index)

        def search(query_embeddings: List[np.ndarray], k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
            if len(query_embeddings) != num_embedding_types:
                raise ValueError(f"Expected {num_embedding_types} embedding types, got {len(query_embeddings)}")

            # Normalize query embeddings
            normalized_queries = []
            for query_emb in query_embeddings:
                norm = np.linalg.norm(query_emb) + 0.001
                normalized_queries.append((query_emb / norm).astype('float32').reshape(1, -1))

            # Initial wide search in each embedding space
            expanded_k = min(k * 2, len(data))  # Search for more candidates initially
            all_candidates = set()
            per_type_results = []

            for i, (index, query) in enumerate(zip(indexes, normalized_queries)):
                distances, indices = index.search(query, expanded_k)
                all_candidates.update(indices[0])
                per_type_results.append((distances[0], indices[0]))

            # Calculate final scores for all candidates
            final_scores = []
            for idx in all_candidates:
                type_distances = []
                close_matches = 0

                # Collect distances for this candidate across all embedding types
                for type_idx, (distances, indices) in enumerate(per_type_results):
                    if idx in indices:
                        dist = distances[np.where(indices == idx)[0][0]]
                        if dist < boost_threshold:
                            how_close = 1
                            while 2 ** how_close < 1e308 and dist < boost_threshold / (2 ** how_close):
                                how_close += 1
                            # increase_func = lambda x: x**2 /2  # sum(list(range(how_close)))
                            increase_func = lambda x: x
                            close_matches += increase_func(how_close)
                        type_distances.append(dist)
                    else:
                        # If this candidate wasn't in top-k for this embedding type,
                        # use a default high distance
                        type_distances.append(default_high_distance)

                # Calculate final score

                if close_matches > 0:
                    # The more close matches, the more we discount other distances
                    discount_factor = 1.0 / close_matches
                    type_distances = np.array(type_distances)
                    closest = min(type_distances)
                    discounted_type_distances = np.maximum(type_distances * discount_factor, closest)
                    # Use the best close distance, plus discounted sum of other distances
                    final_score = sum(discounted_type_distances) / len(discounted_type_distances)
                else:
                    # If no close matches, use regular average
                    final_score = sum(type_distances) / len(type_distances)

                final_scores.append((final_score, idx))

            # Sort and get top k
            final_scores.sort()
            top_k = final_scores[:k]

            # Format results
            result_distances = np.array([[score for score, _ in top_k]])
            result_indices = np.array([[idx for _, idx in top_k]])

            return result_distances, result_indices

        return search

    # Re-verify embedding quality over large dataset -----------------------------------------------------------------
    @staticmethod
    def limit_ann_functions_wrapper(function, data_indexes_list) -> SearchFunction:
        """
        Call ANN function with subset of data
        """
        def indexing_stage(data: EmbeddingsList) -> SearchFunction:
            if not data or not all(data):
                raise ValueError("Data must be non-empty and all entries must have embeddings")

            if isinstance(data, np.ndarray):
                data = data[:data_indexes_list]
            else:
                data = [[row[ind] for ind in data_indexes_list] for row in data]

            resulting_search = function(data)

            def search(query_embeddings: List[np.ndarray], k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
                if isinstance(query_embeddings, np.ndarray):
                    query_embeddings = query_embeddings[data_indexes_list]
                else:
                    query_embeddings = [query_embeddings[ind] for ind in data_indexes_list]
                return resulting_search(query_embeddings, k)

            return search

        return indexing_stage
