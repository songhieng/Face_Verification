import face_recognition
import numpy as np
import pickle
from mtcnn import MTCNN
from PIL import Image
import cv2
import faiss
import imgaug.augmenters as iaa
import os
import gradio as gr

def detect_and_align_face(image_path):
    detector = MTCNN()
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detections = detector.detect_faces(image_rgb)
    
    if len(detections) == 0:
        raise ValueError("No face detected in the image.")
    
    detection = detections[0]
    x, y, width, height = detection['box']
    keypoints = detection['keypoints']
    face = image_rgb[y:y+height, x:x+width]
    
    left_eye = keypoints['left_eye']
    right_eye = keypoints['right_eye']
    delta_x = right_eye[0] - left_eye[0]
    delta_y = right_eye[1] - left_eye[1]
    angle = np.arctan2(delta_y, delta_x) * (180.0 / np.pi)
    
    center = ((x + x + width) // 2, (y + y + height) // 2)
    rot_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)
    aligned_image = cv2.warpAffine(image_rgb, rot_matrix, (image_rgb.shape[1], image_rgb.shape[0]))
    aligned_face = aligned_image[y:y+height, x:x+width]
    
    return Image.fromarray(aligned_face)

def load_encodings(file_path):
    with open(file_path, "rb") as file:
        data = pickle.load(file)
    return np.array(data["encodings"]), data["labels"]

def save_encodings(encodings, labels, file_path):
    data = {"encodings": encodings, "labels": labels}
    with open(file_path, "wb") as file:
        pickle.dump(data, file)

def create_faiss_index(known_encodings):
    dimension = known_encodings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(known_encodings)
    return index

def encode_face(image):
    img_array = np.array(image)
    encodings = face_recognition.face_encodings(img_array)
    return encodings[0] if encodings else None

def augment_image(image, num_augmented=5):
    image = np.array(image)
    aug = iaa.Sequential([
        iaa.Fliplr(0.5),  # horizontal flips
        iaa.Affine(rotate=(-25, 25)),  # rotation
        iaa.AdditiveGaussianNoise(scale=(0, 0.05*255)),  # noise
        iaa.Multiply((0.8, 1.2)),  # brightness
        iaa.GaussianBlur(sigma=(0.0, 1.0))  # blur
    ])
    augmented_images = [Image.fromarray(aug(image=image)) for _ in range(num_augmented)]
    return augmented_images

def update_dataset_with_verified_image(image, encodings_file, label, num_augmented=5):
    known_encodings, known_labels = load_encodings(encodings_file)
    augmented_images = augment_image(image, num_augmented=num_augmented)
    images_to_encode = [image] + augmented_images
    for img in images_to_encode:
        img_array = np.array(img)
        encoding = face_recognition.face_encodings(img_array)
        if encoding:
            known_encodings = np.append(known_encodings, [encoding[0]], axis=0)
            known_labels.append(label)
    save_encodings(known_encodings, known_labels, encodings_file)

def verify_face_with_faiss(image, encodings_file, similarity_threshold=70, num_augmented=5):
    aligned_face = image.convert("RGB")
    target_encodings = face_recognition.face_encodings(np.array(aligned_face))
    
    if not target_encodings:
        return False, "No face detected in the provided image. Please try with another image."
    
    target_encoding = target_encodings[0].reshape(1, -1)
    
    known_encodings, known_labels = load_encodings(encodings_file)
    
    if len(known_encodings) == 0:
        return False, "No known faces in the database. Please add some faces first."
    
    known_encodings = np.array(known_encodings)
    
    index = create_faiss_index(known_encodings)
    
    distances, indices = index.search(target_encoding, 1)
    
    best_match_index = indices[0][0]
    best_similarity_percentage = (1 - distances[0][0]) * 100
    
    is_match = best_similarity_percentage >= similarity_threshold
    
    if is_match:
        matched_label = known_labels[best_match_index]
        update_dataset_with_verified_image(image, encodings_file, matched_label, num_augmented=num_augmented)
        return True, f"Match found: {matched_label}, Similarity: {best_similarity_percentage:.2f}%"
    else:
        return False, "No match found."

# Gradio Interface Function
def gradio_interface(image, similarity_threshold=70):
    if image is None:
        return "Error: No image provided. Please upload an image or take one using the webcam."
    
    encodings_file = "./face_encoding.pkl"
    
    try:
        result, message = verify_face_with_faiss(image, encodings_file, similarity_threshold=similarity_threshold)
        return message
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Gradio Interface Setup
iface = gr.Interface(
    fn=gradio_interface, 
    inputs=[gr.Image(type="pil"), gr.Slider(0, 100, value=70, label="Similarity Threshold")], 
    outputs="text", 
    title="Face Recognition with MTCNN and FAISS",
    description="Upload an image to see if it matches any face in the database."
)

iface.launch(share=True)
