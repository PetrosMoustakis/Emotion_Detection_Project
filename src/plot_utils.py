import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

try:
    from src.config import image_classes, train_dir
    from data_preprocessing_utils import count_images_in_subfolders
except ImportError as e:
    print(f"Error importing modules: {e}")


def display_sample_images(base_folder, displayed_classes, image_name="im0.png"):
    """
    Displays the first image from each class subfolder.
    """
    if not displayed_classes or not base_folder:
        print("Error: base_folder_path or classes_to_display is empty.")
        return
    plt.figure(figsize=(11, 11))
    for i, cls_name in enumerate(displayed_classes):
        plt.subplot(3, 3, i + 1)
        path = os.path.join(base_folder, cls_name, image_name)
        if os.path.exists(path):
            try:
                img = plt.imread(path)
                plt.imshow(img, cmap='gray')
                plt.xlabel(image_classes[i])
            except Exception as e:
                print(f"Could not display image {path}", {e})
        else:
            print(f"Error: This path does not exists {path}")

        plt.xticks([])
        plt.yticks([])

    plt.show()

def plot_class_distribution(labels):
    """
    Plot the distribution of each class in the dataset.
    """
    integer_labels = np.argmax(labels, axis=1)

    plt.figure(figsize=(10, 6))
    sns.countplot(x=integer_labels)
    plt.title('Distribution of Classes')
    plt.xlabel('Class Label')
    plt.ylabel('Number of Images')

    # Set the x-axis ticks to be the actual class names for better readability
    plt.xticks(ticks=range(len(image_classes)), labels=image_classes, rotation=45)

    plt.show()

if __name__ == "__main__":
    print("Plot Utils")
