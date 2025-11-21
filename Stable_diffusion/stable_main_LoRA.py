
"""
Script de fine_tuning de Stable_diffusion avec les adaptaters de la méthode LoRA de HuggingFace

Pour fine_tuner Stable_diffusion on utilisera la lib diffusers qui gère la mécanique de training LoRA


1- Import des librairies
2- loading du modèle pré-entrainé Stable_diffusion v1-5 avec le Pipeline de diffusers
3- loading et processing du dataset
4- configuration des paramètres d'entrainement
5- configuration de LoRA
6- boucle d'entrainement

"""



# ////////////////////////
# 1- Import des librairies
# ////////////////////////

import torch
from pathlib import Path
from diffusers import StableDiffusionImg2ImgPipeline
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import torchvision.transforms as transforms
from peft import get_peft_model, LoraConfig, TaskType
import torch.nn.functional as F
from torchvision import transforms
import csv
from sklearn.model_selection import train_test_split
import random

DEBUG = False


# check l'accès aux GPUs
torch.cuda.is_available()



# ////////////////////////////////////
# 2- Loading et processing du dataset
# ////////////////////////////////////

"""
Objectif :
1- création des couples inp/tar
2- mélange des couples
3- split du dataset en train/val/test
4- création des tensors PyTorch [C,H,W] normalisés entre [0,1]
5- création des dataloaders nécessaires pour LoRA

"""

DATA_DIR = Path("./git/ImageMLProject/Stable_diffusion/dataset_velocity")

input_dir = os.path.join(DATA_DIR, "input")
target_dir = os.path.join(DATA_DIR, "target")
all_files = sorted(os.listdir(input_dir))

# shuffle des couples
random.seed(42)
random.shuffle(all_files)


# split du dataset en train/val/test 80%/10%/10%
train_files, tmp_files = train_test_split(all_files, test_size=0.2, random_state=42)
val_files, test_files = train_test_split(tmp_files, test_size=0.5, random_state=42)

print(f"Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")

output_dir = Path("./LoRA_results")
os.makedirs(output_dir, exist_ok=True)


class VelocityDataset (Dataset):
    def __init__(self, data_dir, file_list=None):
        self.input_dir = os.path.join(data_dir, "input")
        self.target_dir = os.path.join(data_dir, "target")
        self.files = file_list
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        fnames = self.files[idx]
        input_path = os.path.join(self.input_dir, fnames)
        target_path = os.path.join(self.target_dir, fnames)
        # chargement des images .png : loader d'images
        input_image = Image.open(input_path).convert("RGB") # PIL.image (512,512) [0,255]
        target_image = Image.open(target_path).convert("RGB") 
        # transformation des images en np.array
        input_arr = np.array(input_image)
        target_arr = np.array(target_image) # np.ndarray (512,512,3) [0,255]
        # transformation en tensors PyTorch [0,1] (format pour LoRA)
        inp_tensor = torch.from_numpy(input_arr) # torch.tensor [512,512,3] [0,255]
        tar_tensor = torch.from_numpy(target_arr)
        inp_tensor = inp_tensor.permute(2, 0, 1).float() / 255.0 # torch.tensor [3,512,512] [0,1]
        tar_tensor = tar_tensor.permute(2, 0, 1).float() / 255.0 
        return inp_tensor, tar_tensor

"""
possiblité d'utiliser torchvision.transforms :

# ToTensor() :
# 1- conversion PIL.Image -> torch.Tensor
# 2- permute les canaux
# 3- normalise les valeurs entre 0 et 1

transform = transforms.Compose([
    transforms.ConvertImageDtype(torch.float32),
    transforms.ToTensor(),  # convertit automatiquement en C,H,W et normalise entre 0–1
])

inp = transform(Image.open(inp_path).convert("RGB"))
tgt = transform(Image.open(tgt_path).convert("RGB"))

"""
BATCH_SIZE = 1

# loading des datasets
train_dataset = VelocityDataset(DATA_DIR, train_files)
val_dataset = VelocityDataset(DATA_DIR, val_files)
test_dataset = VelocityDataset(DATA_DIR, test_files)

if DEBUG :
    inp, tar = train_dataset[0]
    print(inp.shape, tar.shape, inp.dtype, tar.dtype) # torch.Tensor float32 [3,512,512]
    print(np.min(inp.numpy()), np.max(inp.numpy())) # [0,1]


train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True) # torch.Tensor float32 [1,3,512,512]
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


if DEBUG:
    inp, tar = next(iter(train_loader))
    print(inp.shape, tar.shape, inp.dtype, tar.dtype, type(inp), type(tar))
    # torch.Tensor float32 [1,3,512,512]
    import matplotlib.pyplot as plt
    plt.subplot(1, 2, 1)
    plt.imshow(inp[0].permute(1, 2, 0).cpu())  # tensor -> HWC
    plt.title("Input t")
    plt.subplot(1, 2, 2)
    plt.imshow(tar[0].permute(1, 2, 0).cpu())
    plt.title("Target t+1")
    plt.show()



# //////////////////////////////////
# 3- loading du modèle pré-entrainé
# //////////////////////////////////

"""
Sur le O :

import torch
import transformers
from diffusers import StableDiffusionImg2ImgPipeline

model_id = "runwayml/stable-diffusion-v1-5"
model = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(
    "cuda"
)
model.save_pretrained("./$CCCSCRATCHDIR/git/ImageMLProject/Stable_diffusion")
"""


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model_path = Path(
    "./git/ImageMLProject/Stable_diffusion/SD1-5_local"
)
# charger le modèle avec diffusers et la méthode adaptée pour Image_to_Image
model = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_path, 
    dtype=torch.float16).to(DEVICE)

# récupération de l'unet du modèle
unet = model.unet



# /////////////////////////
# 4- configuration de LoRA
# /////////////////////////


# Configurer LoRA
config = LoraConfig(
    r=8,  # nbre de dimensions latentes : plus r est grand, plus l'adaptation est flexible mais plus il y a de paramètres à entraîner'
    lora_alpha=32,
    target_modules=["to_q", "to_v"],  # couches attention du UNet
    lora_dropout=0.1,
    bias="none"
)



"""
# Ajouter une couche LoRA à ton UNet
model.unet.add_adapter("lora_adapter", rank=8)
model.unet.enable_adapters()
"""



# /////////////////////////////////////////////////
# 5- configuration des paramètres d'entrainements
# /////////////////////////////////////////////////

EPOCHS = 100
LR = 1e-5
# Optimiseur
optimizer = torch.optim.Adam(model.unet.parameters(), lr=LR)
# model
model.unet = get_peft_model(model.unet, config)
unet.train()


# //////////////////////////////////
# 6- boucle d'entrainement
# //////////////////////////////////



# Boucle d'entrainement
"""
Objectif : on va seulement entrainer l'unet
on garde figer le VAE et le CLIP (encoder image et text)

NB : l'unet prend en entrée 
- un latent bruité = sample // format des latents = [B,C,H/8,W/8]
- un timestep (étape de diffusion du bruit)
- des embeddings de texte pour conditionner la prediction

=> il faut construire un pipeline de formatage des inputs :
    - loader les batches
    - convertir les tensors en latents
    - ajouter du bruit
    - préparer les embeddings de texte

dans l'entrainement :
    - faire une prediction bruitée
    - calculer la loss
    - backward + optimizer step

"""

if DEBUG:
    inp, tar = next(iter(train_loader)) # tensor [1,3,512,512]
    inp = inp.to(DEVICE) 
    tar = tar.to(DEVICE)
    # conversion en latents
    latents_input = model.vae.encode(inp).latent_dist.sample() * 0.18215 # tensor [1,4,64,64]
    latents_target = model.vae.encode(tar).latent_dist.sample() * 0.18215
    # ajout du bruit gaussien à différentes timesteps selon un schedule
    timestep = torch.randint(0, model.scheduler.num_train_timesteps, (latents_input.shape[0],), device = DEVICE).long()
    noise = torch.randn_like(latents_input)
    noisy_latents_input = model.scheduler.add_noise(latents_input, noise, timestep)
    # préparation de l'encoding du texte
    prompt = ["scientific visualization of 2D velocity field, next timestep prediction, grayscale simulation"]*BATCH_SIZE
    input_ids = model.tokenizer(prompt, padding="max_length", max_length=model.tokenizer.model_max_length, return_tensors="pt")
    encoder_hidden_states = model.text_encoder(input_ids.input_ids.to(DEVICE))[0]
    # forward pass
    pred = model.unet(noisy_latents_input, timestep, encoder_hidden_states).sample
    # calcul de la perte
    loss = F.l1_loss(pred, latents_target)
    print(loss)
    

losses_list = []
for epoch in range(EPOCHS):
    epoch_loss = 0.0
    unet.train()
    for inp, tar in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"): # affichage d'une barre de progression
        inp, tar = inp.to(DEVICE), tar.to(DEVICE) # envoie sur GPU 
        # encodage en latents
        with torch.no_grad():
            latents_input = model.vae.encode(inp).latent_dist.sample() * 0.18215 # tensor [1,4,64,64]
            latents_target = model.vae.encode(tar).latent_dist.sample() * 0.18215
        # ajout du bruit gaussien
        timestep = torch.randint(
            low=0, 
            high=model.scheduler.num_train_timesteps, 
            size=(latents_input.shape[0],),
            device = DEVICE).long()
        noise = torch.randn_like(latents_input)
        noisy_latents_input = model.scheduler.add_noise(latents_input, noise, timestep)
        # encoding du texte // conditonnement par le texte
        prompt = ["scientific visualization of 2D velocity field, next timestep prediction, grayscale simulation"] * BATCH_SIZE
        inputs = model.tokenizer(prompt, padding="max_length", max_length=model.tokenizer.model_max_length, return_tensors="pt")
        encoder_hidden_states = model.text_encoder(inputs.input_ids.to(DEVICE))[0]
        # unet forward pass
        pred_noise = unet(noisy_latents_input, 
                          timestep, 
                          encoder_hidden_states).sample 
        # calcul de la perte
        loss = F.mse_loss(pred_noise, noise) # MSE error : le modèle apprend à débruiter et non à predire la target
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    losses_list.append(epoch_loss / len(train_loader))
    print(f"Epoch {epoch+1} - Loss: {epoch_loss / len(train_loader):.6f}")





# sauvegarde de la liste des pertes 
with open("loss_history.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "loss"])
    for i, loss in enumerate(losses_list, 1):
        writer.writerow([i, loss])




# ///////////////////
# Sauvegarde du modèle
# //////////////////

model.unet.save_pretrained(output_dir)
model.save_pretrained(output_dir)