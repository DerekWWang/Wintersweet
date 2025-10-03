import pandas as pd
import shutil
from pathlib import Path

def split_training(base_path="/Users/derekzhu/Code/UW_LABMED/bab2", labels_file="data/Labels.csv"):
    """
    Split training data into folders based on labels
    
    Args:
        base_path (str): Base path to the project directory
        labels_file (str): Relative path to the labels CSV file
    """
    labels_path = Path(base_path) / labels_file
    
    if not labels_path.exists():
        print(f"Error: Labels file not found at {labels_path}")
        return
    
    try:
        df = pd.read_csv(labels_path)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
    
    data_dir = Path(base_path) / "data" 
    training_data_dir = data_dir / "core_data"
    
    if not training_data_dir.exists():
        print(f"Error: Training data directory not found at {training_data_dir}")
        return

    labels = df['label'].unique()
    label_dirs = {label: data_dir / label for label in labels}
    meta = {label: 0 for label in labels}
    missing_count = 0

    print(f"Found {len(labels)} unique labels: {list(labels)}")
    print(f"Processing {len(df)} images...")

    # Create output directories if they don't exist
    for label, dir_path in label_dirs.items():
        dir_path.mkdir(exist_ok=True)
        print(f"Created/verified directory: {dir_path}")

    for idx, row in df.iterrows():
        image_id = row['ImageID']
        label = row['label']
        
        source_image = training_data_dir / f"{image_id}.jpg"
        dest_dir = label_dirs[label]

        if source_image.exists():
            try:
                shutil.copy(source_image, dest_dir / f"{image_id}.jpg")
                meta[label] += 1
            except Exception as e:
                print(f"Error copying {source_image}: {e}")
        else:
            missing_count += 1
            print(f"Warning: Image {image_id}.jpg not found in training_data")

    print("\nSplit complete!")
    for label, count in meta.items():
        print(f"  {label}: {count} images copied")
    
    if missing_count > 0:
        print(f"  Missing images: {missing_count}")

if __name__ == "__main__":
    split_training()

