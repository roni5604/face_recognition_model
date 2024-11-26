# scripts/prepare_dataset.py

import os
import shutil
from sklearn.model_selection import train_test_split

def prepare_dataset(dataset_path="data/face-expression-recognition-dataset/images/images", split_ratios=(0.7, 0.15, 0.15)):
    # Paths to the train and validation directories
    train_source = os.path.join(dataset_path, 'train')
    val_source = os.path.join(dataset_path, 'validation')
    combined_dataset = os.path.join(dataset_path, 'combined')

    # Combine train and validation data into one directory
    if not os.path.exists(combined_dataset):
        os.makedirs(combined_dataset)
        # List emotion categories, excluding hidden files
        emotions = [d for d in os.listdir(train_source) if os.path.isdir(os.path.join(train_source, d)) and not d.startswith('.')]
        for emotion in emotions:
            emotion_combined_dir = os.path.join(combined_dataset, emotion)
            os.makedirs(emotion_combined_dir, exist_ok=True)
            # Copy training images
            train_emotion_dir = os.path.join(train_source, emotion)
            for img in os.listdir(train_emotion_dir):
                if not img.startswith('.'):
                    shutil.copy(os.path.join(train_emotion_dir, img), emotion_combined_dir)
            # Copy validation images
            val_emotion_dir = os.path.join(val_source, emotion)
            for img in os.listdir(val_emotion_dir):
                if not img.startswith('.'):
                    shutil.copy(os.path.join(val_emotion_dir, img), emotion_combined_dir)

    # Now split the combined dataset into new train, validation, and test sets
    emotions = [d for d in os.listdir(combined_dataset) if os.path.isdir(os.path.join(combined_dataset, d)) and not d.startswith('.')]
    train_dir = os.path.join(dataset_path, 'new_train')
    val_dir = os.path.join(dataset_path, 'new_validation')
    test_dir = os.path.join(dataset_path, 'test')

    # Create new directories or clear them if they already exist
    for directory in [train_dir, val_dir, test_dir]:
        if os.path.exists(directory):
            shutil.rmtree(directory)
        os.makedirs(directory, exist_ok=True)

    for emotion in emotions:
        emotion_dir = os.path.join(combined_dataset, emotion)
        images = [img for img in os.listdir(emotion_dir) if os.path.isfile(os.path.join(emotion_dir, img)) and not img.startswith('.')]

        if len(images) < 3:
            print(f"Not enough images in {emotion_dir} to split. Skipping...")
            continue

        train_imgs, temp_imgs = train_test_split(images, test_size=(1 - split_ratios[0]), random_state=42)
        val_imgs, test_imgs = train_test_split(temp_imgs, test_size=(split_ratios[2] / (split_ratios[1] + split_ratios[2])), random_state=42)

        # Move images to new directories
        for imgs, subset_dir in zip([train_imgs, val_imgs, test_imgs], [train_dir, val_dir, test_dir]):
            emotion_subset_dir = os.path.join(subset_dir, emotion)
            os.makedirs(emotion_subset_dir, exist_ok=True)
            for img in imgs:
                shutil.move(os.path.join(emotion_dir, img), os.path.join(emotion_subset_dir, img))

    # Remove the combined dataset directory
    shutil.rmtree(combined_dataset)

if __name__ == "__main__":
    prepare_dataset()
