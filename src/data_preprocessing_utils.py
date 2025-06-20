import os
import cv2
import tensorflow as tf
import numpy as np
from tensorflow.keras import utils
from src.config import image_classes


def count_images_in_subfolders(base_folder):
    """
    Counts the number of images into every subfolder of base folder
    :param base_folder: The path to the directory containing class subfolders e.g (train, test)
    :return: A dictionary where keys are subfolder names (class names) and values the counts of files in the subfolder
    """
    counts = {}

    if not os.path.isdir(base_folder):
        print(f"Error: Directory not found - {base_folder}")

    for subfolder in os.listdir(base_folder):
        path = os.path.join(base_folder, subfolder)

        if not os.path.isdir(path):
            print(f"Error: {path} is not a directory")
        else:
            counts[subfolder] = len(os.listdir(path))

    return counts


def load_and_preprocess_data(data_dir, img_size, color_mode='grayscale'):
    """
    Load and preprocess the data.
    :param data_dir: The directory containing the data
    :param img_size: The size of the image
    :param color_mode: The color mode of the image
    :return: The normalized array of images with range [0,1] and the labels into one-hot vectors.
    """
    images=[]
    labels=[]

    num_classes = len(image_classes)

    for idx, folder in enumerate(image_classes):
        folder_path = os.path.join(data_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        for img in os.listdir(folder_path):
            try:
                img_path = os.path.join(folder_path, img)

                if color_mode == 'grayscale':
                    temp_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                else:
                    temp_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                    temp_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB)

                if temp_img is not None:
                    temp_img = cv2.resize(temp_img, img_size)
                    images.append(temp_img)
                    labels.append(idx)
                else:
                    print(f"{folder_path} has no image {img_path}")
            except Exception as e:
                 print(f"Error loading image {img} in {folder}: {e}")

    images = np.array(images)

    if color_mode == 'grayscale':
        images = images.reshape(images.shape[0], img_size[0], img_size[1], 1)

    images = images.astype('float32') / 255.0

    labels = utils.to_categorical(labels, num_classes = num_classes)

    return images, labels

if __name__ == "__main__":
    print("Data Preprocessing Utils")
