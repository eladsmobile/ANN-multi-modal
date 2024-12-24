import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import random
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_tensor
from torchvision.datasets import CIFAR100
from time import time


# Image transformation functions ------------------------------------------------------------------------------------
def apply_rotation_changes(image):
    """Apply different levels of rotation"""
    transforms_list = [
        transforms.RandomRotation(degrees=(0, 0)),  # No rotation
        transforms.RandomRotation(degrees=(2, 2)),  # Slight rotation
        transforms.RandomRotation(degrees=(90, 90)),  # Moderate rotation
        transforms.RandomRotation(degrees=(60, 60))  # Severe rotation
    ]
    return [transform(image) for transform in transforms_list]


def apply_brightness_changes(image):
    """Apply different levels of brightness adjustment"""
    transforms_list = [
        transforms.ColorJitter(brightness=0),  # Original
        transforms.ColorJitter(brightness=0.2),  # Slight brightness change
        transforms.ColorJitter(brightness=0.5),  # Moderate brightness change
        transforms.ColorJitter(brightness=0.8)  # Severe brightness change
    ]
    return [transform(image) for transform in transforms_list]


def apply_noise_changes(image):
    """Apply different levels of Gaussian noise"""

    def add_gaussian_noise(img, std):
        img_array = np.array(img)
        noise = np.random.normal(0, std, img_array.shape)
        noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_img)

    return [
        image,  # Original
        add_gaussian_noise(image, 10),  # Light noise
        add_gaussian_noise(image, 25),  # Medium noise
        add_gaussian_noise(image, 50)  # Heavy noise
    ]


def apply_blur_changes(image):
    """Apply different levels of Gaussian blur"""
    transforms_list = [
        transforms.GaussianBlur(kernel_size=1, sigma=0.1),  # No blur
        transforms.GaussianBlur(kernel_size=3, sigma=1.0),  # Light blur
        transforms.GaussianBlur(kernel_size=5, sigma=2.0),  # Medium blur
        transforms.GaussianBlur(kernel_size=7, sigma=3.0)  # Heavy blur
    ]
    return [transform(image) for transform in transforms_list]


def apply_crop_changes(image):
    """Apply different levels of cropping."""
    width, height = image.size

    # Function to perform random crop near edges
    def random_crop_near_edge(image, crop_size):
        left = random.choice([0, width - crop_size[0]])
        top = random.choice([0, height - crop_size[1]])
        return transforms.functional.crop(image, top, left, crop_size[1], crop_size[0])

    transforms_list = [
        transforms.Lambda(lambda x: x),  # No crop
        transforms.CenterCrop((30, 30)),  # Center crop 30x30
        transforms.CenterCrop((16, 16)),  # Center crop 16x16
        transforms.Lambda(lambda x: random_crop_near_edge(x, (16, 16)))  # Random crop 16x16 near the edges
    ]

    return [transform(image) for transform in transforms_list]


# Visualization functions for transformations and the dataset --------------------------------------------------------
def get_sample_images_from_cifar100(n_images=25, seed=None):
    """Get n_images unique images from CIFAR100 dataset"""
    # Load CIFAR100 dataset
    dataset = CIFAR100(root='./data', train=True, download=True)

    # Randomly select n_images unique indices
    random.seed(seed if seed else time())
    selected_indices = random.sample(range(len(dataset)), n_images)

    # Get the images
    images = []
    for idx in selected_indices:
        img, _ = dataset[idx]
        images.append(img)

    return images


def create_transformation_dataset(images):
    """Create dataset with all transformation levels for each image"""
    all_transformed_images = {}

    transformation_functions = {"rotation": apply_rotation_changes,
                                "brightness": apply_brightness_changes,
                                "noise": apply_noise_changes,
                                "blur": apply_blur_changes,
                                "crop": apply_crop_changes  # changes size
                                }
    for transformation_name, transform_func in transformation_functions.items():
        result = []
        for idx, image in enumerate(images):
            # Select transformation function based on image index
            transformed_images = transform_func(image)
            result.extend(transformed_images)
        all_transformed_images[transformation_name] = result

    return all_transformed_images


def display_image_grid(images, n_rows=5, n_cols=5, figsize=(10, 10)):
    """Display images in a grid"""
    plt.figure(figsize=figsize)
    for idx, image in enumerate(images):
        if idx >= n_rows * n_cols:
            break
        plt.subplot(n_rows, n_cols, idx + 1)
        plt.imshow(image)
        plt.axis('off')
    plt.tight_layout()
    plt.show()


