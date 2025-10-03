from pathlib import Path
import sklearn.model_selection
import pandas as pd

def split_patients(base_path="/Code/DL/bbosis/Hongyuan-Babesiosis", labels_file="data/Labels.csv"):
    ptid2images = {}
    ptid2labels = {}

    df = pd.read_csv(f"{base_path}/{labels_file}")
    for _, row in df.iterrows():
        ptid = row['PtID']
        img = row['ImageID']
        lab = row['label']
        if ptid not in ptid2images:
            ptid2images[ptid] = []
            ptid2labels[ptid] = []
        ptid2images[ptid].append(img)
        ptid2labels[ptid].append(lab)

    ptids = df['PtID'].unique().tolist()
    train_ids, test_ids = sklearn.model_selection.train_test_split(ptids, test_size=0.2, random_state=42)

    data_dir = f"{base_path}/data"
    data_dir = Path(data_dir)
    (data_dir/"train").mkdir(exist_ok=True)
    (data_dir/"test").mkdir(exist_ok=True)
    
    training_set = []
    for i in train_ids:
        images = ptid2images[i]
        labels = ptid2labels[i]
        for img, lab in zip(images, labels):
            training_set.append({"img_path": f"core_data/{img}.jpg", "label": lab})
        
    testing_set = []
    for i in test_ids:
        images = ptid2images[i]
        labels = ptid2labels[i]
        for img, lab in zip(images, labels):
            testing_set.append({"img_path": f"core_data/{img}.jpg", "label": lab})

    pd.DataFrame(training_set).to_csv(data_dir/"train/training.csv", index=False)
    pd.DataFrame(testing_set).to_csv(data_dir/"test/testing.csv", index=False)


    print(f"Found {len(ptid2images)} patients with {len(df)} images")

if __name__ == "__main__":
    split_training()