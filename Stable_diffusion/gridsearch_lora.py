
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
from itertools import product
import copy
import matplotlib.pyplot as plt
import json
from stable_dataset import generate_dataset
from diffusers.models.attention_processor import LoRAAttnProcessor

DEBUG = False


# check l'accès aux GPUs
torch.cuda.is_available()

metadata_path = generate_dataset(
    dataset_path=Path("./git/ImageMLProject/Datasets/CFD_Dataset/film_3_512x512.npy"),
    data_dir="./git/ImageMLProject/Stable_diffusion/dataset_velocity",
    use_prompt=True,
    resize_shape=(512,512)
)


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
"""

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
"""


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
DATA_DIR = os.path.join("./git/ImageMLProject/Stable_diffusion/dataset_velocity")



# récupération des données
with open(metadata_path, "r") as f:
    metadata = [json.loads(line) for line in f]

# shuffle des couples
random.seed(42)
random.shuffle(metadata)

# split du dataset en train/val/test
train_files, tmp_files = train_test_split(metadata, test_size=0.2, random_state=42)
val_files, test_files = train_test_split(tmp_files, test_size=0.5, random_state=42)

# sauvegarde des datasets
def save_jsonl(meta_list, save_path):
    with open(save_path, "w") as f:
        for entry in meta_list:
            f.write(json.dumps(entry))
            f.write("\n")

save_jsonl(train_files, os.path.join(DATA_DIR, "train_metadata.jsonl"))
save_jsonl(val_files, os.path.join(DATA_DIR, "val_metadata.jsonl"))
save_jsonl(test_files, os.path.join(DATA_DIR, "test_metadata.jsonl"))


# loading des datasets
train_dataset = VelocityDataset(DATA_DIR, metadata_path=os.path.join(DATA_DIR, "train_metadata.jsonl"))
val_dataset = VelocityDataset(DATA_DIR, metadata_path=os.path.join(DATA_DIR, "val_metadata.jsonl"))
test_dataset = VelocityDataset(DATA_DIR, metadata_path=os.path.join(DATA_DIR, "test_metadata.jsonl"))

if DEBUG :
    for i in range(5):  # les 5 premiers éléments
        inp, tar, caption = train_dataset[i]
        print(f"Sample {i}:")
        print(f"  Input tensor shape : {inp.shape}, dtype={inp.dtype}, min={np.min(inp.numpy())}, max={np.max(inp.numpy())}")
        print(f"  Target tensor shape: {tar.shape}, dtype={tar.dtype}")
        # torch.Tensor float32 [3,512,512] [0,1]
        print(f"  Caption  : '{caption}'")


train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True) # torch.Tensor float32 [1,3,512,512]
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


if DEBUG:
    inp, tar, caption = next(iter(train_loader))
    print(inp.shape, inp.dtype, type(inp), np.min(inp.numpy()), np.max(inp.numpy()), f"caption : {caption}")
    # torch.Tensor float32 [1,3,512,512] [0,1]
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

model = StableDiffusionImg2ImgPipeline.from_pretrained(model_path) 
model = model.to(dtype=torch.float16).to(DEVICE)

model.text_encoder.eval()
for p in model.text_encoder.parameters():
    p.requires_grad_(False)

# /////////////////////////////////////////////
# 4- définition de la fonction d'entrainement
# /////////////////////////////////////////////

output_dir = Path("./LoRA_results")
os.makedirs(output_dir, exist_ok=True)


def train_lora (model, train_loader, val_loader, lora_config,
                epochs=100, lr=1e-5,
                device=DEVICE, patience=5):
    # integration des adaptaters LoRA dans l'UNet
    # Intégrer LoRA sur le UNet
    model.unet = get_peft_model(model.unet, lora_config)
    model.unet.to(device, dtype=torch.float32)
    optimizer = torch.optim.Adam(model.unet.parameters(), lr=lr)
    train_loss_list = []
    val_loss_list = []
    best_val_loss = float('inf')
    counter = 0
    # boucle d'entrainement
    for epoch in range(epochs):
        model.unet.train()
        train_loss = 0.0
        for inp, tar, caption in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"): # affichage d'une barre de progression
            inp = inp.to(device, torch.float32) # envoie sur GPU 
            # encodage en latents
            with torch.no_grad():
                latents = model.vae.encode(inp).latent_dist.sample()
                latents = latents * 0.18215
                latents = latents.to(device, torch.float32)
            # ajout du bruit gaussien
            bsz = latents.shape[0]
            timesteps = torch.randint(
                0, model.scheduler.num_train_timesteps,
                (bsz,), device=device, dtype=torch.long
            )
            noise = torch.randn_like(latents)
            noisy_latents = model.scheduler.add_noise(latents, noise, timesteps)
            noisy_latents = noisy_latents.to(torch.float32)
            # encoding du texte (si CFD caption)
            if caption is not None:
                inputs_text = model.tokenizer(
                    list(caption),
                    padding="max_length",
                    truncation=True,
                    max_length=model.tokenizer.model_max_length,
                    return_tensors="pt"
                )
                with torch.no_grad():
                    encoder_hidden_states = model.text_encoder(
                        inputs_text.input_ids.to(device)
                    )[0]
                encoder_hidden_states = encoder_hidden_states.to(device, torch.float32)
            else:
                encoder_hidden_states = None
            # unet forward pass
            noise_pred = model.unet(
                noisy_latents, timesteps, 
                encoder_hidden_states).sample
            # calcul de la perte
            loss = F.mse_loss(noise_pred, noise) # MSE error : le modèle apprend à débruiter et non à predire la target
            # backward pass
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
                inp = inp.to(device, torch.float32) # envoie sur GPU 
                # encodage en latents
                latents = model.vae.encode(inp).latent_dist.sample()
                latents = latents * 0.18215
                latents = latents.to(device, torch.float32)
                # ajout du bruit gaussien
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0, model.scheduler.num_train_timesteps,
                    (bsz,), device=device, dtype=torch.long
                )
                noise = torch.randn_like(latents)
                noisy_latents = model.scheduler.add_noise(latents, noise, timesteps)
                noisy_latents = noisy_latents.to(torch.float32)
                if caption is not None:
                    inputs_text = model.tokenizer(
                        list(caption),
                        padding="max_length",
                        truncation=True,
                        max_length=model.tokenizer.model_max_length,
                        return_tensors="pt"
                    )
                    encoder_hidden_states = model.text_encoder(
                        inputs_text.input_ids.to(device)
                    )[0]
                    encoder_hidden_states = encoder_hidden_states.to(device, torch.float32)
                else:
                    encoder_hidden_states = None
                noise_pred = model.unet(
                    noisy_latents, timesteps, 
                    encoder_hidden_states).sample
                loss = F.mse_loss(noise_pred, noise)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_loss_list.append(val_loss)
        # early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            # Sauvegarde du meilleur LoRA (partie UNet uniquement)
            model.unet.save_pretrained(output_dir)
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        print(f"Epoch {epoch+1} | Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f}")

    return train_loss_list, val_loss_list, best_val_loss     



# //////////////////////////////////////////////////
# 5- définition de la fonction de Grid Search LoRA
# //////////////////////////////////////////////////



def run_experiment(model, train_loader, val_loader, 
                   save_dir, 
                   patience=3, epochs=100, lr=1e-4, device=DEVICE):
    os.makedirs(save_dir, exist_ok=True)
    # grille des configs LoRA
    r_values = [4, 8, 16, 32]
    lora_alpha_values = [16, 32, 64]
    dropout_values = [0.0, 0.1, 0.2]
    batch_size = [1, 2, 4]
    epochs = [10,30,50]
    lr_values = [1e-4, 2e-5, 1e-5]
    use_prompt = [True, False]
    configs = list(product(r_values,
                    lora_alpha_values,
                    dropout_values,
                    batch_size,
                    epochs,
                    lr_values))
    results = []
    for (r, alpha, dropout, batch_size, epochs, lr_val) in configs:
        print(f"Training with LoRA config: r={r}, alpha={alpha}, dropout={dropout}, epochs={epochs}, lr={lr_values}")
        exp_name = f"r{r}_alpha{alpha}_dropout{dropout}_batchsize{batch_size}_epochs{epochs}"
        exp_dir = os.path.join(output_dir, exp_name)
        os.makedirs(exp_dir, exist_ok=True)
        # Configurer LoRA
        lora_config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            lora_dropout=dropout,
            bias="none")
        # clone du modèle pour chaque config
        model_copy = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float32
        ).to(device)
        # lancement de l'entrainement
        train_loss_list, val_loss_list, best_val_loss = train_lora(model_copy, train_loader, val_loader,
                                                lora_config=lora_config, 
                                                epochs=epochs, lr=lr_val, 
                                                patience=patience, device=device)
        # enregistrement des resultas
        results.append({
            "config": exp_name,
            "best_val_loss": best_val_loss,
            "val_loss_history": val_loss_list,
            "train_loss_history": train_loss_list
        })
        for name, loss_list in [("val", val_loss_list), ("train", train_loss_list)]:
            csv_path = os.path.join(exp_dir, f"{name}_loss_history.csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "loss"])
                for i, loss in enumerate(loss_list, start=1):
                    writer.writerow([i, loss])
            # Sauvegarde courbe des pertes
            plt.figure()
            plt.plot(train_loss_list, label="Training loss", color="blue", linewidth=2)
            plt.plot(val_loss_list, label="Validation loss", color="red", linestyle="--", linewidth=2)
            plt.xlabel("Epoch")
            plt.ylabel("MSE Loss")
            plt.legend()
            plt.title(f"Losses - {exp_name}")
            plt.tight_layout()
            plt.savefig(os.path.join(exp_dir, "losses_curve.png"), dpi=300)
            plt.close()

    return results


# //////////////////////////////////////////
# 6- entrainement avec grid search LoRA
# //////////////////////////////////////////

if __name__ == "__main__":
    results = run_experiment(model, train_loader, val_loader, 
                             output_dir, lr=1e-4, 
                             patience=3, device=DEVICE)
    print("All experiments done")
# tri et sélection du meilleur modèle
    best = sorted(results, key=lambda x: x["best_val_loss"])[0] # tri par meilleure perte de validation
    print(f"Best LoRA config: {best['config']} with Val MSE: {best['best_val_loss']:.6f}")
    best_lora_dir = output_dir / best['config']
    with open(os.path.join(best_lora_dir, "best_lora_config.txt"), "w") as f:
        f.write(str(best_lora_dir))
