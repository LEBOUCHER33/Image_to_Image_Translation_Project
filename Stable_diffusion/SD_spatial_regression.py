
"""
Script de fine_tuning de Stable_diffusion avec les adaptaters de la méthode LoRA de HuggingFace.

Pour fine_tuner Stable_diffusion on utilisera la lib diffusers qui gère la mécanique de training LoRA.

Dans ce script on modifira le rôle de debruiteur de l'UNet pour l'adapter à une fonction de regression spatiale :

Workflow du pipeline :
1- encodage de l'image d'entrée avec le VAE => latent_inp
2- encodage de l'image cible avec le VAE => latent_tar
3- génération d'un latent de prediction avec l'UNet => latent_pred
4- calcul de la loss : MAE(latent_pred, latent_tar)
5- generation de l'image predite : VAE.decode(latent_pred)


Workflow du script :
1- Import des librairies
2- loading du modèle pré-entrainé Stable_diffusion v1-5 avec le Pipeline de diffusers
3- loading et processing du dataset
4- configuration des paramètres d'entrainement
5- configuration de LoRA
6- boucle d'entrainement

"""



# //////////////////////////////////////////////////////////
# 1- Import des librairies
# //////////////////////////////////////////////////////////

import torch
from pathlib import Path
from diffusers import StableDiffusionImg2ImgPipeline
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
from peft import get_peft_model, LoraConfig, TaskType
import torch.nn.functional as F
import csv
import random
from itertools import product
import matplotlib.pyplot as plt
import json

DEBUG = False


# check l'accès aux GPUs
torch.cuda.is_available()

#////////////////////////////////////////////////////////////
# 2- définition du dataset 
#////////////////////////////////////////////////////////////

# on repart du même dataset 

DATA_DIR = os.path.join("./git/ImageMLProject/Stable_diffusion/dataset_velocity")

class VelocityDataset (Dataset):
    def __init__(self, data_dir, metadata_path=None):
        self.data_dir = data_dir
        self.input_dir = os.path.join(data_dir, "input")
        self.target_dir = os.path.join(data_dir, "target")
        if metadata_path is None:
            metadata_path = os.path.join(data_dir, "metadata.jsonl")
        with open(metadata_path, "r") as f:
            self.metadata = [json.loads(line) for line in f]
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        sample = self.metadata[idx]
        input_path = os.path.join(self.data_dir, sample["input"])
        target_path = os.path.join(self.data_dir, sample["target"])
        caption = sample.get("caption", "")
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
        return inp_tensor, tar_tensor, caption




# loading des datasets

train_dataset = VelocityDataset(DATA_DIR, metadata_path=os.path.join(DATA_DIR, "train_metadata.jsonl"))
val_dataset = VelocityDataset(DATA_DIR, metadata_path=os.path.join(DATA_DIR, "val_metadata.jsonl"))
test_dataset = VelocityDataset(DATA_DIR, metadata_path=os.path.join(DATA_DIR, "test_metadata.jsonl"))

if DEBUG :
    for i in range(5):  # les 5 premiers éléments
        inp, tar = train_dataset[i]
        print(f"Sample {i}:")
        print(f"  Input tensor shape : {inp.shape}, dtype={inp.dtype}, min={np.min(inp.numpy())}, max={np.max(inp.numpy())}")
        print(f"  Target tensor shape: {tar.shape}, dtype={tar.dtype}")
        # torch.Tensor float32 [3,512,512] [0,1]


# génération des DataLoaders avec différents batch_size


def make_dataloaders (train_dataset, val_dataset, test_dataset, batch_size=1):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # torch.Tensor float32 [1,3,512,512]
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader




if DEBUG:
    train_loader, val_loader, test_loader = make_dataloaders(train_dataset, val_dataset, test_dataset)
    inp, tar = next(iter(train_loader))
    print(inp.shape, inp.dtype, type(inp), np.min(inp.numpy()), np.max(inp.numpy()))
    # torch.Tensor float32 [1,3,512,512] [0,1]
    import matplotlib.pyplot as plt
    plt.subplot(1, 2, 1)
    plt.imshow(inp[0].permute(1, 2, 0).cpu())  # tensor -> HWC
    plt.title("Input t")
    plt.subplot(1, 2, 2)
    plt.imshow(tar[0].permute(1, 2, 0).cpu())
    plt.title("Target t+1")
    plt.show()



# /////////////////////////////////////////////////////////////
# 3- loading du modèle pré-entrainé
# /////////////////////////////////////////////////////////////

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

model = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_path, 
    dtype=torch.float16).to(DEVICE)



# /////////////////////////////////////////////////////////////////////
# 4- définition de la fonction d'entrainement
# ////////////////////////////////////////////////////////////////////

output_dir = Path("./LoRA_results_regression_space")
os.makedirs(output_dir, exist_ok=True)


def train_lora_regression (model, train_loader, val_loader, lora_config,
                epochs=100, lr=1e-5,
                device=DEVICE, patience=5):
    """
    _Summary_: fonction pour generer des predictions avec LoRA
    _Args_: 
        - model : modèle pré-entrainé
        - train_loader : dataloader d'entrainement
        - val_loader : dataloader de validation
        - lora_config : configuration de LoRA
        - epochs : nombre d'epochs d'entrainement
        - lr : learning rate
    _Returns_:
        - train_loss_list : liste des pertes d'entrainement
        - val_loss_list : liste des pertes de validation
        - best_val_loss : perte de validation la plus basse
    
    """
    # integration des adaptaters LoRA dans l'UNet
    model.unet = get_peft_model(model.unet, lora_config)
    optimizer = torch.optim.Adam(model.unet.parameters(), lr=lr)
    mae_loss = torch.nn.L1Loss()
    train_loss_list, val_loss_list = [], []
    best_val_loss = float('inf')
    counter = 0
    # boucle d'entrainement
    for epoch in range(epochs):
        train_loss = 0.0
        model.unet.train()
        for inp, tar, caption in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"): # affichage d'une barre de progression
            inp, tar = inp.to(DEVICE), tar.to(DEVICE) # envoie sur GPU 
            # encodage en latents inputs et targets
            with torch.no_grad():
                latents_input = model.vae.encode(inp).latent_dist.sample() * 0.18215 # tensor [1,4,64,64]
                latents_target = model.vae.encode(tar).latent_dist.sample() * 0.18215
            # création d'un embedding neutre via CLIP
            inputs_text = model.tokenizer([""],
                                        padding="max_length",
                                        max_length=model.tokenizer.model_max_length,
                                        return_tensors="pt")
            text_embeddings = model.text_encoder(inputs_text.input_ids.to(device))[0]
            # génération du latent_pred avec l'UNet sans ajout de bruit
            latents_pred = model.unet(latents_input, 
                                      timestep=torch.tensor([0],device=device), 
                                      encoder_hidden_states=text_embeddings).sample
            # calcul de la perte
            loss = mae_loss(latents_pred, latents_target)
            # backward pass = retropropagation classique
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_loss_list.append(train_loss)
        # validation
        # on évalue la perte sur le dataset de validation
        model.unet.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inp, tar, caption in val_loader:
                inp, tar = inp.to(device), tar.to(device) # envoie sur GPU 
                # encodage en latents
                latents_input = model.vae.encode(inp).latent_dist.sample() * 0.18215 # tensor [1,4,64,64]
                latents_target = model.vae.encode(tar).latent_dist.sample() * 0.18215
                # encodage d'un texte neutre
                inputs_text = model.tokenizer([""],
                                              padding="max_length",
                                              max_length=model.tokenizer.model_max_length,
                                              return_tensors="pt")
                text_embeddings = model.text_encoder(inputs_text.input_ids.to(device))[0]
                # génération du latent_pred avec l'UNet sans ajout de bruit
                latents_pred = model.unet(latents_input, 
                                          timestep=torch.tensor([0],device=device), 
                                          encoder_hidden_states=text_embeddings
                                         ).sample
                loss = mae_loss(latents_pred, latents_target)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_loss_list.append(val_loss)
        # early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            # sauvegarde du meilleur modèle
            model.unet.save_pretrained(output_dir)
            model.save_pretrained(output_dir)
            with open(os.path.join(output_dir, "best_lora_config.txt"), "w") as f:
                f.write(str(lora_config))
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        # affichage des pertes
        print(f"Epoch {epoch+1} | Train MAE: {train_loss:.6f} | Val MAE: {val_loss:.6f}")
    return train_loss_list, val_loss_list, best_val_loss        



# //////////////////////////////////////////////////
# 5- définition de la fonction de Grid Search LoRA
# //////////////////////////////////////////////////



def run_experiment(model, train_dataset, val_dataset, test_dataset,
                   save_dir, 
                   patience=5, epochs=100, lr=1e-5, device=DEVICE):
    os.makedirs(save_dir, exist_ok=True)
    # grille des configs LoRA
    r_values = [4, 8, 16, 32]
    lora_alpha_values = [16, 32, 64]
    dropout_values = [0.0, 0.1, 0.2]
    batch_size = [1,2,4,6,8,10]
    configs =  list(product(r_values, lora_alpha_values, dropout_values, batch_size))
    results = []
    for (r, alpha, dropout, batch_size) in configs:
        print(f"Training with LoRA config: r={r}, alpha={alpha}, dropout={dropout}")
        exp_name = f"r{r}_alpha{alpha}_dropout{dropout}_batchsize{batch_size}"
        exp_dir = Path(save_dir) / exp_name
        os.makedirs(exp_dir, exist_ok=True)
        # création dataloaders
        train_loader, val_loader, test_loader = make_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=batch_size)
        # Configurer LoRA
        lora_config = LoraConfig(
            r=r,  # nbre de dimensions latentes : plus r est grand, plus    
            lora_alpha=alpha,
            target_modules=["to_q", "to_v"],  # couches attention du UNet
            lora_dropout=dropout,   
            bias="none"
        )
        # clone du modèle pour chaque config
        model_copy = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_path, 
            dtype=torch.float16).to(device)
        # lancement de l'entrainement
        train_loss_list, val_loss_list, best_val_loss = train_lora_regression(model_copy, train_loader, val_loader, 
                                                lora_config=lora_config, 
                                                epochs=epochs, lr=lr, 
                                                patience=patience, device=device)
        # enregistrement des resultas
        results.append({
            "config": exp_name,
            "best_val_loss": best_val_loss,
            "val_loss_history": val_loss_list,
            "train_loss_history": train_loss_list
        })
        # sauvegarde de la liste des pertes
        with open(os.path.join(save_dir, "val_loss_history.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "loss"])
            for i, loss in enumerate(val_loss_list, 1):
                writer.writerow([i, loss])
        with open(os.path.join(save_dir, "train_loss_history.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "loss"])
            for i, loss in enumerate(train_loss_list, 1):
                writer.writerow([i, loss])
        # sauvegarde de la courbe des pertes
        plt.figure()
        plt.plot(val_loss_list, label="Validation loss", color="red", linewidth=2, linestyle="--")
        plt.plot(train_loss_list, label="Training loss", color="blue", linewidth=2, linestyle="-")
        plt.xlabel("Epoch") 
        plt.ylabel("MSE Loss")
        plt.legend()
        plt.savefig(os.path.join(save_dir, "losses_curve.png"))
        plt.close()
    return results


# //////////////////////////////////////////
# 6- entrainement avec grid search LoRA
# //////////////////////////////////////////

if __name__ == "__main__":
    results = run_experiment(model, train_dataset, val_dataset, test_dataset,
                             output_dir, epochs=100, lr=1e-5, 
                             patience=10, device=DEVICE)
# tri et sélection du meilleur modèle
    best = sorted(results, key=lambda x: x["best_val_loss"])[0] # tri par meilleure perte de validation
    print(f"Best LoRA config: {best['config']} with Val MSE: {best['best_val_loss']:.6f}")
    best_lora_dir = output_dir / best['config']
    with open(os.path.join(best_lora_dir, "best_lora_config.txt"), "w") as f:
        f.write(str(best_lora_dir))
