from torchvision import transforms
import torch
import timm
import pandas as pd
import numpy as np
from PIL import Image, ImageOps
from slide_util import slide_to_tiles


SMALL_CHECKPOINT = "hf_hub:Snarcy/RedDino-small"
BASE_CHECKPOINT = "hf_hub:Snarcy/RedDino-base"

device = "cuda" if torch.cuda.is_available() else "cpu"

model = timm.create_model(BASE_CHECKPOINT, pretrained=True)
model.eval()
model.to(device)
print("Model loaded to", device)

def create_bag(img_path, model) -> torch.Tensor: 
    def preprocess(image):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        return transform(image).unsqueeze(0)

    tiles = slide_to_tiles(np.asarray(ImageOps.exif_transpose(Image.open(img_path)).convert("RGB")), 224, 0, -1)
    
    bag_embeddings = torch.empty((0, model.num_features), device=device)

    for tile in tiles:
        img = preprocess(tile).to(device)  # Move input to the correct device
        with torch.no_grad():  # Disable grad for efficiency
            output_features = model(img)
        bag_embeddings = torch.cat((bag_embeddings, output_features), dim=0)


    print(bag_embeddings.shape)
    print(f"Processed {img_path}, bag size: {len(bag_embeddings)}")
    return bag_embeddings

def labels2bags(labels_df):
    embeddings = []
    labels = []

    for index, row in labels_df.iterrows():
        image_id = row['img_path']
        label = row['label']

        bag = create_bag(f"C:/Code/DL/bbosis/Hongyuan-Babesiosis/data/{image_id}", model)
        embeddings.append(bag.cpu())  # Store bag on CPU to save GPU memory
        labels.append(1 if label == "BABSP" else 0)

        print(f"Processed {image_id} with label {label}, labels {labels[-1]}")

    return embeddings, torch.tensor(labels)


e, l = labels2bags(pd.read_csv("C:/Code/DL/bbosis/Hongyuan-Babesiosis/data/training.csv"))
torch.save(e, "C:/Code/DL/bbosis/Hongyuan-Babesiosis/data/tensors/embeddings.pt")
torch.save(l, "C:/Code/DL/bbosis/Hongyuan-Babesiosis/data/tensors/labels.pt")