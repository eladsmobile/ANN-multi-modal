import numpy as np
import matplotlib.pyplot as plt
import random
import requests
import zipfile
from tqdm import tqdm
from scipy.spatial.distance import cosine, cdist
import tensorflow as tf
import os


def download_tiny_imagenet(base_dir):
    """
    Downloads and extracts Tiny ImageNet dataset
    """
    os.makedirs(base_dir, exist_ok=True)
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    zip_path = os.path.join(base_dir, "tiny-imagenet-200.zip")

    if not os.path.exists(zip_path):
        print("Downloading Tiny ImageNet...")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))

        with open(zip_path, 'wb') as f:
            for data in tqdm(response.iter_content(chunk_size=1024), total=total_size // 1024):
                f.write(data)

    if not os.path.exists(os.path.join(base_dir, "tiny-imagenet-200")):
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(base_dir)


def get_similar_image_pairs_tiny(num_pairs, pair_size=2, base_dir="./dataset", similarity_threshold=0.7, image_size=64):
    """
    Returns pairs of similar Tiny ImageNet images where pairs are from the same class,
    with additional similarity check for better matching.

    Args:
        num_pairs (int): Number of pairs to return
        base_dir (str): Directory to store/load dataset
        similarity_threshold (float): Minimum similarity for paired images
        image_size (int): Size to resize images to (Tiny ImageNet images are 64x64)

    Returns:
        numpy array: Selected images array of shape (num_pairs*2, image_size, image_size, 3)
        list: Class names for each pair
    """
    # Download and extract dataset if needed
    download_tiny_imagenet(base_dir)

    # Create feature extractor for similarity checks
    feature_extractor = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights='imagenet',
        input_shape=(image_size, image_size, 3),
        pooling='avg'
    )

    # Path to training images
    train_dir = os.path.join(base_dir, "tiny-imagenet-200", "train")

    # Dictionary to store images by class
    class_images = {}
    class_features = {}
    selected_images = []
    selected_class_names = []
    used_classes = set()

    def load_and_preprocess_image(image_path):
        """Load and preprocess image for feature extraction"""
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (image_size, image_size))
        img = tf.cast(img, tf.float32)
        return img, tf.keras.applications.mobilenet_v2.preprocess_input(img)

    # Load class data
    print("Loading dataset...")
    class_dirs = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]

    class_dirs = random.sample(class_dirs, num_pairs)

    for class_dir in tqdm(class_dirs):
        class_path = os.path.join(train_dir, class_dir)
        image_files = [f for f in os.listdir(os.path.join(class_path, "images"))
                       if f.endswith('.JPEG')][:min(500, 10 + pair_size)]  # Take up to 10 images per class

        if len(image_files) >= pair_size:  # Only include classes with at least 2 images
            class_images[class_dir] = []
            class_features[class_dir] = []

            for img_file in image_files:
                img_path = os.path.join(class_path, "images", img_file)
                original_img, processed_img = load_and_preprocess_image(img_path)
                features = feature_extractor.predict(tf.expand_dims(processed_img, 0), verbose=0)

                class_images[class_dir].append(original_img)
                class_features[class_dir].append(features.flatten())

    print("Finding similar pairs...")
    """
    for _ in tqdm(range(num_pairs)):
        # Find an unused class
        available_classes = [c for c in class_images.keys()
                             if c not in used_classes and len(class_images[c]) >= pair_size]

        if not available_classes:
            raise ValueError(f"Not enough unique classes with sufficient images. "
                             f"Only found {len(selected_images) // pair_size} pairs.")

        # Pick a random unused class
        chosen_class = np.random.choice(available_classes)
        """
    for chosen_class in class_images.keys():
        if len(class_images[chosen_class]) <= pair_size:
            raise ValueError(f"Not enough unique classes with sufficient images. "
                             f"Only found {len(selected_images) // pair_size} pairs.")
        used_classes.add(chosen_class)

        # Get features for this class
        class_feat = class_features[chosen_class]
        class_imgs = class_images[chosen_class]

        # Pick first image randomly
        first_idx = np.random.randint(len(class_imgs))
        first_features = class_feat[first_idx]

        # Find most similar image
        similarities = [1 - cosine(first_features, feat) for feat in class_feat]
        similarities[first_idx] = -1  # Exclude the same image

        selected_class_names.append(chosen_class)
        selected_images.extend([class_imgs[first_idx].numpy()])
        # second_idx = np.argmax(similarities)
        pair_inxs = np.argpartition(similarities, pair_size - 1)[-(pair_size - 1):]
        # Add the pair to our selection

        for second_idx in pair_inxs:
            selected_images.extend([class_imgs[second_idx].numpy()])

    return np.array(selected_images), selected_class_names


def visualize_tiny_imagenet_pairs(images, class_names, pair_size=2):
    """
    Helper function to display the image pairs side by side with their class names

    Args:
        images (numpy array): Array of images to display
        class_names (list): List of class names for each pair
    """

    num_pairs = len(images) // pair_size
    fig, axes = plt.subplots(num_pairs, pair_size, figsize=(10, 5 * num_pairs))

    if num_pairs == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_pairs):
        for j in range(pair_size):
            idx = i * pair_size + j
            axes[i, j].imshow(images[idx].astype('uint8'))
            axes[i, j].axis('off')
            if j == 0:
                axes[i, j].set_title(f'Class: {class_names[i]}')

    plt.tight_layout()
    plt.show()


def display_images_with_labels_nparray(image1, image2, image3):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image1.astype('uint8'))
    axes[0].set_title("Query image")
    axes[0].axis('off')

    axes[1].imshow(image2.astype('uint8'))
    axes[1].set_title("Agreed similar")
    axes[1].axis('off')

    axes[2].imshow(image3.astype('uint8'))
    axes[2].set_title("Proposed alternative similar")
    axes[2].axis('off')

    plt.show()
