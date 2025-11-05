"""
Script pour le processing du dataset de simulation CFD pour fine_tuning de Stable_diffusion :
On veut un dataset organisé avec
    - inputs (format .png)
    - targets (format .png)
    - metadata.jsonl

- dataset : film de simulation de CFD = fichier .npy de dimension (990,512,512)

- objectif : processing du fichier .npy en images RGB 512x512: 
    - load du fichier .npy
    - création des paires input/target avec un offset de 1

"""


import numpy as np
from pathlib import Path
import os
from PIL import Image
import json

DEBUG = False

# 1- définir le dataset

dataset_name = "film_3_512x512.npy"
dataset_path = Path("./git/ImageMLProject/Datasets/CFD_Dataset/") / dataset_name
dataset_path = dataset_path.resolve()


if DEBUG:
    print(dataset_path)
    data = np.load(dataset_path)
    print(data.shape) # (990,512,512)
    print(type(data)) # np.darray
    print(np.min(data), np.max(data)) # [0,255]
    assert len(data.shape) == 3


# 2- processing du dataset

data_dir = "./git/ImageMLProject/Stable_diffusion/dataset_velocity"

"""
# création des dossiers des data
os.makedirs(data_dir, exist_ok=True)
os.makedirs(os.path.join(data_dir, "input"), exist_ok=True)
os.makedirs(os.path.join(data_dir, "target"), exist_ok=True)


# load du fichier .npy
data = np.load(dataset_path)
nb_paires = data.shape[0] - 1
metadata = []
use_prompt = False


# processing des data

resize_shape = (512,512)

for i in range (nb_paires):
    inputs = data[i] # np.ndarray (512,512) [0,255]
    targets = data[i+1]
    # conversion en image 8-bit
    inp_img = Image.fromarray(inputs.astype(np.uint8)).convert("L") # PIL.image (512,512) [0,255] niveaux de gris
    tar_img = Image.fromarray(targets.astype(np.uint8)).convert("L")
    # resize si necessaire
    inp_img = inp_img.resize(resize_shape)
    tar_img = tar_img.resize(resize_shape)
    # sauvegarde
    inp_path = f"input/{i}.png"
    tar_path = f"target/{i}.png"
    inp_img.save(os.path.join(data_dir, inp_path))
    tar_img.save(os.path.join(data_dir, tar_path))
    # metadata.jsonl
    caption_text = "CFD velocity 2D t to t+1" if use_prompt else ""
    metadata.append({
        "input": inp_path,
        "target": tar_path,
        "caption": caption_text
    })


# sauvegarde du fichier jsonl
with open(os.path.join(data_dir, "metadata.jsonl"), "w") as f:
    for entry in metadata:
        json.dump(entry, f)
        f.write("\n")



"""
# /////////////////////////////////////////////////
# création d'une fonction pour générer le dataset
# /////////////////////////////////////////////////

def generate_dataset (dataset_path, data_dir, use_prompt=False, resize_shape=(512,512)):
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(os.path.join(data_dir, "input"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "target"), exist_ok=True)
    data = np.load(dataset_path)
    nb_paires = data.shape[0] - 1
    metadata = []
    for i in range (nb_paires):
        inputs = data[i] # np.ndarray (512,512) [0,255]
        targets = data[i+1]
        # conversion en image 8-bit
        inp_img = Image.fromarray(inputs.astype(np.uint8)).convert("L") # PIL.image (512,512) [0,255]
        tar_img = Image.fromarray(targets.astype(np.uint8)).convert("L")
        # resize si necessaire
        inp_img = inp_img.resize(resize_shape)
        tar_img = tar_img.resize(resize_shape)
        # sauvegarde
        inp_path = f"input/{i}.png"
        tar_path = f"target/{i}.png"
        inp_img.save(os.path.join(data_dir, inp_path))
        tar_img.save(os.path.join(data_dir, tar_path))
        # prompt
        caption_text = "CFD velocity 2D t to t+1" if use_prompt else ""
        metadata.append({
            "input": inp_path,
            "target": tar_path,
            "caption": caption_text
        })
    # sauvegarde du fichier jsonl
    metadata_path = os.path.join(data_dir, "metadata.jsonl")
    with open(os.path.join(data_dir, "metadata.jsonl"), "w") as f:
        for entry in metadata:  
            f.write(json.dumps(entry))
            f.write("\n")
    return metadata_path





 




