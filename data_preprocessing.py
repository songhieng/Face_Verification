import os
import pickle
import numpy as np
import imgaug.augmenters as iaa
from PIL import Image
import face_recognition

def augment_image(image, num_augmented=5):
    """
    Apply data augmentation to an image.
    
    Parameters:
    image (PIL.Image): The image to augment.
    num_augmented (int): Number of augmented images to generate.
    
    Returns:
    List[PIL.Image]: List of augmented images.
    """
    image = np.array(image)

    # Define a sequence of augmentation techniques
    aug = iaa.Sequential([
        iaa.Fliplr(0.5),  # horizontal flips
        iaa.Affine(rotate=(-25, 25)),  # rotation
        iaa.AdditiveGaussianNoise(scale=(0, 0.05*255)),  # noise
        iaa.Multiply((0.8, 1.2)),  # brightness
        iaa.GaussianBlur(sigma=(0.0, 1.0))  # blur
    ])

    # Generate augmented images
    augmented_images = [Image.fromarray(aug(image=image)) for _ in range(num_augmented)]
    return augmented_images

def preprocess_and_save_augmented_encodings(image_dir, output_file, num_augmented=5):
    known_encodings = []
    known_labels = []

    # List all files in the image directory
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    for image_path in image_paths:
        # Load the original image
        original_image = Image.open(image_path).convert("RGB")  # Ensure the image is in RGB format

        # Augment the image
        augmented_images = augment_image(original_image, num_augmented=num_augmented)

        # Include the original image in the list of images to encode
        images_to_encode = [original_image] + augmented_images

        for img in images_to_encode:
            img_array = np.array(img)
            # Encode the face
            encoding = face_recognition.face_encodings(img_array)[0]
            
            # Store the encoding and the corresponding label
            known_encodings.append(encoding)
            known_labels.append(image_path)  # You can use a more descriptive label here

    # Save encodings and labels to a file
    data = {"encodings": known_encodings, "labels": known_labels}
    with open(output_file, "wb") as file:
        pickle.dump(data, file)

# Example usage:
image_dir = "train"  # Directory containing all the training images
output_file = "face_encoding.pkl"
preprocess_and_save_augmented_encodings(image_dir, output_file, num_augmented=5)
