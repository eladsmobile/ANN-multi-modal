import numpy as np
import cv2
from scipy.spatial.distance import cosine, cdist
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.cluster import KMeans
from torchvision.transforms.functional import to_tensor
from skimage.feature import graycomatrix, graycoprops
import torchvision.models as PTmodels
from skimage.measure import shannon_entropy
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
import os
import warnings
import time
from tqdm import tqdm

warnings.filterwarnings("ignore", message="KMeans is known to have a memory leak")

os.environ["OMP_NUM_THREADS"] = "4"

# Global vars for some functions
n_features = 100
sift = cv2.SIFT_create(nfeatures=n_features)


# Simple auxiliary, commonly used methods for embedding
class EmbeddingFunctions:
    def __init__(self, init=True):
        self.embedding_functions = {}

        # loading pretrained models from cnn_embedding, resnet18_embedding, CLIP embedding functions
        # cnn_embedding (vgg16 version)
        self.vgg16 = PTmodels.vgg16(weights=PTmodels.VGG16_Weights.IMAGENET1K_V1)
        self.vgg16 = torch.nn.Sequential(*list(self.vgg16.children())[:-1])
        self.vgg16.eval()
        # cnn_embedding (resnet50 version)
        self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.resnet50 = torch.nn.Sequential(*list(self.resnet50.children())[:-1])
        self.resnet50.eval()
        # resnet18_embedding
        self.resnet18 = PTmodels.resnet18(weights=PTmodels.ResNet18_Weights.IMAGENET1K_V1)
        self.resnet18.eval()
        # common transform for the above models
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),])
        # semantic_concept_embedding
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        # object_composition_embedding
        self.faster_resnet50 = PTmodels.detection.fasterrcnn_resnet50_fpn(
            weights=PTmodels.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        self.faster_resnet50.eval()

        if init:
            self.embedding_functions["simple_color_histogram_embedding"] = self.simple_color_histogram_embedding
            self.embedding_functions["texture_embedding"] = self.texture_embedding
            self.embedding_functions["composition_rules_embedding"] = self.composition_rules_embedding
            self.embedding_functions["scene_complexity_embedding"] = self.scene_complexity_embedding
            self.embedding_functions["downsample_embedding"] = self.downsample_embedding
            self.embedding_functions["edge_histogram_embedding"] = self.edge_histogram_embedding
            self.embedding_functions["fft_embedding"] = self.fft_embedding
            self.embedding_functions["sift_embedding"] = self.sift_embedding
            self.embedding_functions["cnn_embedding"] = self.cnn_embedding
            self.embedding_functions["resnet18_embedding"] = self.resnet18_embedding
            self.embedding_functions["object_composition_embedding"] = self.object_composition_embedding
            self.embedding_functions["semantic_concept_embedding"] = self.semantic_concept_embedding
            self.embedding_functions["color_palette_embedding"] = self.color_palette_embedding

    # Quick embedding methods --------------------------------------------------------------------------------------------
    @staticmethod
    def simple_color_histogram_embedding(img, bins=32):
        """
        Compute a simple color histogram embedding

        Args:
        img (PIL.Image or torch.Tensor): Input image
        bins (int): Number of bins for color histogram

        Returns:
        np.ndarray: Flattened color histogram
        """
        if not isinstance(img, np.ndarray):
            if not isinstance(img, Image.Image):
                img = transforms.ToPILImage()(img)
            img = np.array(img)

        # Compute histogram for each color channel
        hist_r = np.histogram(img[:, :, 0], bins=bins, range=[0, 256])[0]
        hist_g = np.histogram(img[:, :, 1], bins=bins, range=[0, 256])[0]
        hist_b = np.histogram(img[:, :, 2], bins=bins, range=[0, 256])[0]

        # Normalize and concatenate
        hist = np.concatenate([
            hist_r / np.sum(hist_r),
            hist_g / np.sum(hist_g),
            hist_b / np.sum(hist_b)
        ])

        return hist

    @staticmethod
    def texture_embedding(img, distances=(1,), angles=(0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)):
        """
        Generate texture embedding using Gray Level Co-occurrence Matrix (GLCM).

        Args:
        image_path (str): Path to the input image
        distances (list): List of pixel pair distance offsets
        angles (list): List of pixel pair angles in radians

        Returns:
        np.array: Texture features
        """
        # Load and prepare image
        if not isinstance(img, np.ndarray):
            img = img.convert('L')  # Convert to grayscale
            img = np.array(img)
        if len(img.shape) > 2:
            img = np.array(img).reshape(-1, 3)
        if np.issubdtype(img.dtype, np.floating):
            img = (img * 255).astype(np.uint8)
        # Compute GLCM
        glcm = graycomatrix(img, distances=distances, angles=angles,
                            levels=256, symmetric=True, normed=True)

        # Compute GLCM properties
        contrast = graycoprops(glcm, 'contrast')
        dissimilarity = graycoprops(glcm, 'dissimilarity')
        homogeneity = graycoprops(glcm, 'homogeneity')
        energy = graycoprops(glcm, 'energy')
        correlation = graycoprops(glcm, 'correlation')

        # Combine features
        features = np.hstack([contrast, dissimilarity, homogeneity, energy, correlation])
        max_ = np.max(features)
        features = np.hstack([features / max_, [[np.log(max_)]]])
        return features.flatten()

    @staticmethod
    def composition_rules_embedding(img):  # image_path):
        """
        Generate embedding based on basic photography composition rules.

        Args:
        image_path (str): Path to the input image

        Returns:
        np.array: Composition features
        """
        # img = np.array(Image.open(image_path).convert('RGB'))
        if not isinstance(img, np.ndarray):
            img = np.array(img.convert('RGB'))
        h, w = img.shape[:2]

        # Rule of thirds points
        third_h = h // 3
        third_w = w // 3

        # Calculate feature importance at rule of thirds intersections
        thirds_points = [
            img[third_h, third_w],
            img[third_h, 2 * third_w],
            img[2 * third_h, third_w],
            img[2 * third_h, 2 * third_w]
        ]

        # Center weight
        center_region = img[h // 3:2 * h // 3, w // 3:2 * w // 3]
        center_weight = np.mean(center_region)

        # Convert to features
        thirds_features = np.mean(thirds_points, axis=1)

        return np.concatenate([thirds_features.flatten(), [center_weight]]) / 2 ** 5

    @staticmethod
    def scene_complexity_embedding(img):  # image_path):
        """
        Generate embedding based on image complexity metrics.

        Args:
        image_path (str): Path to the input image

        Returns:
        np.array: Complexity features
        """
        # img = np.array(Image.open(image_path).convert('L'))
        if not isinstance(img, np.ndarray):
            img = np.array(Image.convert('L'))

        # Calculate entropy
        entropy = shannon_entropy(img)

        # Calculate frequency domain features
        f_transform = np.fft.fft2(img)
        f_spectrum = np.abs(np.fft.fftshift(f_transform))
        freq_energy = np.sum(f_spectrum)

        # Calculate number of unique intensity values
        unique_intensities = len(np.unique(img))

        # Calculate local variance
        local_var = np.std(img)

        return np.array([entropy / 3, np.sqrt(freq_energy) / 1000, unique_intensities / 100, local_var / 40])

    @staticmethod
    def downsample_embedding(image, size=(8, 8)):
        """
        Downsample the image to an even smaller resolution and use the pixel values as an embedding.

        """
        small_image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
        return small_image.flatten() / 255

    @staticmethod
    def edge_histogram_embedding(image, expected_number_of_pixel_per_edge=20):
        """
        Use edge detection (e.g., Sobel filter) and create a histogram of edge orientations.
        """
        image = image.astype(np.uint8)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        try:
            edges = cv2.Canny(gray_image, 100, 200)
        except Exception as e:
            print(image)
            raise e
        hist, _ = np.histogram(np.where(edges), bins=32)
        return hist / expected_number_of_pixel_per_edge

    @staticmethod
    def fft_embedding(image):
        "Use FFT to transform the image and compute statistics from the frequency domain"
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        f = np.fft.fft2(gray_image)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift + 0.001))
        return magnitude_spectrum.flatten()[:128] / 90  # Take the first 128 values

    @staticmethod
    def sift_embedding(image_array, n_features=n_features, sift=sift):  # image_path
        """
        Args:
        image_path (str): Path to the input image
        n_features (int): Maximum number of features to extract

        Returns:
        np.array: Fixed-length feature vector combining descriptor information
        """
        # Read image in grayscale
        # img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # if not isinstance(img, np.ndarray):
        #    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        image_array = np.array(image_array)
        if image_array is None or image_array.size == 0:
            print("Image array is empty")
            return np.zeros(128, dtype=np.float32)  # SIFT descriptors are 128-dimensional
        # Ensure the image array is of type uint8
        if image_array.dtype != np.uint8:
            image_array = image_array.astype(np.uint8)
        # Check the shape of the image array
        if len(image_array.shape) != 3 or image_array.shape[2] != 3:
            print("Image array has incorrect shape")
            return np.zeros(128, dtype=np.float32)

        gray_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        # Check if the grayscale conversion was successful
        if gray_image is None or gray_image.dtype != np.uint8:
            print("Grayscale image is empty or has incorrect depth")
            return np.zeros(128, dtype=np.float32)

        # Detect keypoints and compute descriptors
        keypoints, descriptors = sift.detectAndCompute(gray_image, None)

        if descriptors is None:  # Return zero vector if no features found
            return np.zeros(128, dtype=np.float32)  # SIFT descriptors are 128-dimensional
        # Ensure we have a fixed-length output
        if len(keypoints) < n_features:  # Pad with zeros if we found fewer features than requested
            padding = np.zeros((n_features - len(keypoints), 128), dtype=np.float32)
            descriptors = np.vstack((descriptors, padding))
        elif len(keypoints) > n_features:  # Take only the first n_features if we found more
            descriptors = descriptors[:n_features]  # Sum the descriptors vertically and normalize
        summed_descriptors = np.sum(descriptors, axis=0)
        normalized_descriptors = summed_descriptors / np.linalg.norm(summed_descriptors)
        return normalized_descriptors

    # Slower embedding methods ---------------------------------------------------------------------------------------
    def cnn_embedding(self, img, model_name='vgg16'):
        """
        Classic CNN embedding using vgg16.
        Generate CNN features using a pre-trained model.

        Args:
        image_path (str): Path to the input image
        model_name (str): Name of the pre-trained model to use

        Returns:
        np.array: Feature vector
        """
        # Load pre-trained model
        if model_name == 'resnet50':
            model = self.resnet50
        elif model_name == 'vgg16':
            model = self.vgg16
        else:
            raise ValueError("Unsupported model name")

        # img = Image.open(image_path).convert('RGB')
        # Ensure input is PIL Image
        if not isinstance(img, Image.Image):
            img = transforms.ToPILImage()(img)

        img_t = self.transform(img)
        batch_t = torch.unsqueeze(img_t, 0)

        # Generate embedding
        with torch.no_grad():
            embedding = model(batch_t)

        return embedding.numpy().flatten()

    def resnet18_embedding(self, img, layer='avgpool'):
        """
        Faster CNN embedding using resnet18.
        Compute ResNet18 embedding for an image.

        Args:
        img (PIL.Image or torch.Tensor): Input image
        layer (str): Layer to extract embedding from

        Returns:
        np.ndarray: Flattened embedding vector
        """
        # Load pretrained ResNet18
        model = self.resnet18

        # if not isinstance(img, np.ndarray):
        # Ensure input is PIL Image
        if not isinstance(img, Image.Image):
            img = transforms.ToPILImage()(img)

            # Preprocess image
        input_tensor = self.transform(img).unsqueeze(0)

        # Extract features
        with torch.no_grad():
            if layer == 'avgpool':
                # Use average pooling layer
                features = model.avgpool(
                    model.layer4(model.layer3(model.layer2(model.layer1(model.conv1(input_tensor))))))
            elif layer == 'fc':
                # Use final fully connected layer
                features = model.fc(model.avgpool(
                    model.layer4(model.layer3(model.layer2(model.layer1(model.conv1(input_tensor))))).flatten(1)))

        return features.squeeze().numpy().flatten()

    def object_composition_embedding(self, img, boost_weight=10):
        """
        Generate object composition embedding using a pre-trained Faster R-CNN model.

        Args:
        image_path (str): Path to the input image
        threshold (float): Confidence threshold for object detection

        Returns:
        dict: Object composition (class labels and their counts)
        """
        const_number_of_class_in_fasterrcnn_resnet50_fpn = 91

        # Prepare image
        img_tensor = to_tensor(img).unsqueeze(0)

        # Perform object detection
        with torch.no_grad():
            prediction = self.faster_resnet50(img_tensor)

        # Process predictions
        labels = prediction[0]['labels'].numpy()
        scores = prediction[0]['scores'].numpy()

        composition = [0] * const_number_of_class_in_fasterrcnn_resnet50_fpn
        for u, c in zip(labels, scores):
            composition[u] = c * boost_weight

        return np.array(composition)

    def semantic_concept_embedding(self, image):
        """
        Generate embedding based on high-level semantic concepts using CLIP.

        Args:
        image_path (str): Path to the input image

        Returns:
        np.array: Semantic concept features
        """

        # Prepare image
        inputs = self.clip_processor(images=image, return_tensors="pt", padding=True)

        # Generate embedding
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)

        return image_features.numpy()[0]

    @staticmethod
    def color_palette_embedding(img, n_colors=5):
        """
        Generate color palette embedding using K-means clustering.

        Args:
        image_path (str): Path to the input image
        n_colors (int): Number of colors in the palette

        Returns:
        np.array: Color palette embedding
        """
        # Load and prepare image
        # img = Image.open(image_path).convert('RGB')
        # if not isinstance(img, np.ndarray):
        if len(img.shape) > 2:
            img = np.array(img).reshape(-1, 3)

        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_colors, random_state=42)
        kmeans.fit(img)

        # Get color palette and normalize
        palette = kmeans.cluster_centers_.astype(int)
        normalized_palette = palette / 255.0

        return normalized_palette.flatten()

    # Functions for the embedding testing ----------------------------------------------------------------------------
    @staticmethod
    def measure_embedding_times(image_list: Image.Image, embedding_functions: dict, num_examples):
        results = {}
        for name, func in embedding_functions.items():
            start_time = time.time()
            for image in image_list[:num_examples]:
                func(np.array(image))
                if time.time() - start_time > num_examples:
                    break
            end_time = time.time()
            print(f"Time: {round(end_time - start_time, 4)}s\tFunction: {name}")
            results[name] = end_time - start_time
        return results

    @staticmethod
    def compute_embedding_scores(embeddings, start=0.0):
        """
        Compute scores for transformed images based on their embedding distances.
        We have a 4n-sized set of n transformed images - each image has 4 representations in the set.
        Each representation is one of the 4 levels of transformation:
            1. Not transformed (original)
            2. Slightly transformed
            3. Moderately transformed
            4. Heavily transformed
        The goal is to examine how significantly the transformation affects the rank of the transformed image.
        The rank here is the place in the set sorted by cosine similarity with the original image embedding.
        For each image, we compute the final score using the ranks of the 4 transformation levels of the image.
        (See the implementation of the scoring at the end of the function).
        The idea is as follows:
            1. Take the original image embedding.
            2. Sort the set according to the cosine similarity to the original image embedding.
            3. Compute the rank for each of the 4 transformations of the image.
            4. Use the scoring formula to compute the final score.

        Args:
        embeddings (torch.Tensor or np.ndarray): 100 image embeddings

        Returns:
        list: Scores for each set of 4 transformed images
        """
        # Ensure embeddings are numpy array
        if torch.is_tensor(embeddings):
            embeddings = embeddings.numpy()

        # Number of sets (25 original images * 4 transformation groups)
        num_sets = len(embeddings) // 4  # 25
        scores = []
        mean_ranks = {}
        for set_idx in range(num_sets):
            # Select the base (original) image embedding
            base_index = set_idx * 4

            # Select the current set of 4 embeddings
            current_set_embeddings = embeddings[base_index:base_index + 4]
            # base_embedding is the embedding of the original, non-transformed image
            base_embedding = current_set_embeddings[0]

            # Compute distances between base embedding and all 100 embeddings
            distances = cdist([base_embedding], embeddings, metric='cosine')[0]

            # Compute ranks for each transformation
            ranks = []
            sorted_indices = np.argsort(distances)
            # Find the rank of the original images in this set
            for i in range(base_index, base_index + 4):
                ranks.append(np.where(sorted_indices == i)[0][0])
                print(f"\rRun time: {round(time.time() - start, 2)}", end="")

            # Advanced scoring method
            # Exponential penalty for higher ranks with non-linear decay
            score = (
                    4 * np.exp(-ranks[0] / 10) +  # Original image
                    3 * np.exp(-ranks[1] / 20) +  # Slightly transformed
                    2 * np.exp(-ranks[2] / 30) +  # Moderately transformed
                    1 * np.exp(-ranks[3] / 40)  # Heavily transformed
            )

            scores.append(score)
            # mean_ranks.append(round(np.mean(ranks), 4))
            mean_ranks[set_idx] = ranks

        return scores, mean_ranks

    def comprehensive_embedding_analysis(self, image_transformations, embedding_functions, verbose=True):
        """
        Analyze embedding performance across different transformation types and embedding methods

        Args:
        image_transformations (dict): {transform_name: [100 images]}
        embedding_functions (dict): {embedding_name: callable embedding function}
        verbose (bool): Whether to print detailed results

        Returns:
        dict: Comprehensive analysis results
        """
        # Final results storage
        analysis_results = {}
        print(f"Comparing embedding performance and score across different transformation types and embedding methods, on the sample set of {len(image_transformations)} images")
        # Iterate through each embedding method
        for embed_name, embed_func in embedding_functions.items():
            start = time.time()
            print(f"\nMethod: {embed_name}")
            # Store results for this embedding method
            embed_results = {}

            # Iterate through each transformation type
            for transform_name, images in image_transformations.items():
                # Convert images to numpy array of embeddings
                try:
                    embeddings = np.array([embed_func(np.array(img)) for img in images])
                except Exception as e:
                    print(f"Error embedding {transform_name} with {embed_name}: {e}")
                    continue

                # Compute embedding scores
                try:
                    scores, _ = self.compute_embedding_scores(embeddings, start=start)

                    # Compute statistics
                    mean_score = np.mean(scores)
                    std_score = np.std(scores)

                    # Store results
                    embed_results[transform_name] = {
                        # 'scores': scores,
                        'mean_score': round(mean_score, 5),
                        'std_score': round(std_score, 5),
                    }
                except Exception as e:
                    print(f"Error computing scores for {transform_name} with {embed_name}: {e}")

            # Store embedding method results
            analysis_results[embed_name] = embed_results

        # Verbose printing
        if verbose:
            print("\n\n===== Embedding Analysis Results =====")
            for embed_name, embed_results in analysis_results.items():
                print(f"\n{embed_name} Embedding Analysis:")
                for transform_name, result in embed_results.items():
                    print(f"  {transform_name}:")
                    print(f"    Mean Score: {result['mean_score']:.4f}")
                    print(f"    Std Dev:    {result['std_score']:.4f}")

        return analysis_results
