import os
import pandas as pd
import torch
from os import path
from PIL import Image
from torchvision.transforms import ToTensor
import swifter

import config

device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')
image_model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True).to(device)
MAIN_DIR = './data/images'


def img_to_features(panoid):
    img_path = path.abspath(path.join(MAIN_DIR, panoid + '.jpg'))
    feature_path = path.abspath(path.join(MAIN_DIR, 'features', panoid + '.pt'))
    if path.exists(feature_path):
        return 

    image = Image.open(img_path).convert("RGB")
    image = ToTensor()(image).to(device).unsqueeze(0)
    image_features = image_model(image)
    torch.save(image_features, feature_path)


if __name__ == '__main__':
    os.chdir('../')
    node_file = config.paths['node']
    ds = pd.read_csv(node_file,
                     names=["panoid", "heading", "lat", "lon"])

    ds.panoid.swifter.apply(img_to_features)

    print (f"Written {ds.shape[0]} image features.")

