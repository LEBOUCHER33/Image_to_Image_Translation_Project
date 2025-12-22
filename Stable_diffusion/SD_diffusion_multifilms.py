
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
from collections import defaultdict
import pickle

DEBUG = False


# check l'accès aux GPUs
torch.cuda.is_available()



# ////////////////////////////////////////
# 2- Loading et processing du dataset
# ////////////////////////////////////////

"""
Objectif :
1- création des couples inp/tar
2- mélange des couples
3- split du dataset en train/val/test
4- création des tensors PyTorch [C,H,W] normalisés entre [0,1]
5- création des dataloaders nécessaires pour LoRA

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
        film_id = sample["film_id"]
        frame_idx = int(sample["frame_idx"]) 
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
        return inp_tensor, tar_tensor, caption, film_id, frame_idx




# ////////////////////////////////////////////////////
# chargement des films dans un unique dataset 
# ////////////////////////////////////////////////////


all_metadata = []
films_metadata = {} 
DATA_DIR = os.path.join("./git/ImageMLProject/Stable_diffusion/dataset_velocity_geometry")
os.makedirs(DATA_DIR, exist_ok=True)


# concaténation des films

list_films = [74] #[74, 174, 65, 67]
for film_id in list_films:
    npy_path = Path(f"./git/ImageMLProject/Datasets/CFD_Dataset/film_{film_id}_512x512.npy")
    metadata_path = generate_dataset(
        dataset_path=npy_path,
        data_dir=DATA_DIR,
        use_prompt=True,
        resize_shape=(512,512)
    )
    with open(metadata_path, "r") as f:
        metadata = [json.loads(line) for line in f]
        for m in metadata:
            m["film_id"] =  film_id  
        films_metadata[film_id] = metadata


# shuffle des couples dans un même film

global_train = []
global_val   = []
global_test  = []

for film_id, meta_list in films_metadata.items():
    random.seed(42)
    random.shuffle(all_metadata)
    # split du dataset en train/val/test
    train_files, tmp_files = train_test_split(meta_list, test_size=0.2, random_state=42)
    val_files, test_files = train_test_split(tmp_files, test_size=0.5, random_state=42)
    # concat global
    global_train.extend(train_files)
    global_val.extend(val_files)
    global_test.extend(test_files)


"""
global_train = [{'input': 'input/435.png', 'target': 'target/435.png', 'caption': 'CFD simulation of a 2D velocity field at timestep 1, fluid flow visualization, scientific data, heatmap style, spatial regression target', 'film_id': 64}, 
{'input': 'input/102.png', 'target': 'target/102.png', 'caption': 'CFD simulation of a 2D velocity field at timestep 1, fluid flow visualization, scientific data, heatmap style, spatial regression target', 'film_id': 64}]
"""

# sauvegarde des datasets
def save_jsonl(meta_list, save_path):
    with open(save_path, "w") as f:
        for entry in meta_list:
            entry_copy = entry.copy()  # ne pas modifier l'original
            # extraire le numéro de frame depuis le chemin 'input/XXX.png'
            input_filename = os.path.basename(entry['input'])  # "136.png"
            frame_idx = int(os.path.splitext(input_filename)[0])  # 136
            entry_copy['frame_idx'] = frame_idx
            f.write(json.dumps(entry_copy) + "\n")

save_jsonl(global_train, os.path.join(DATA_DIR, "train_metadata.jsonl"))
save_jsonl(global_val, os.path.join(DATA_DIR, "val_metadata.jsonl"))
save_jsonl(global_test, os.path.join(DATA_DIR, "test_metadata.jsonl"))


# loading des datasets
train_dataset = VelocityDataset(DATA_DIR, metadata_path=os.path.join(DATA_DIR, "train_metadata.jsonl"))
val_dataset = VelocityDataset(DATA_DIR, metadata_path=os.path.join(DATA_DIR, "val_metadata.jsonl"))
test_dataset = VelocityDataset(DATA_DIR, metadata_path=os.path.join(DATA_DIR, "test_metadata.jsonl"))

if DEBUG :
    for i in range(5):  # les 5 premiers éléments
        inp, tar, caption, film_ids, sample_ids = train_dataset[i]
        print(f"Sample {i}:")
        print(f"film_id : {film_ids}")
        print(f"  Input tensor shape : {inp.shape}, dtype={inp.dtype}, min={np.min(inp.numpy())}, max={np.max(inp.numpy())}")
        print(f"  Target tensor shape: {tar.shape}, dtype={tar.dtype}")
        # torch.Tensor float32 [3,512,512] [0,1]
        print(f"  Caption  : '{caption}'")


BATCH_SIZE = 1

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True) # torch.Tensor float32 [1,3,512,512]
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)


if DEBUG:
    inp, tar, caption, film_ids, sample_ids = next(iter(train_loader))
    print(inp.shape, inp.dtype, type(inp), 
    np.min(inp.numpy()), np.max(inp.numpy()), f"caption : {caption}",
    f"film_id : {film_ids}")
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

output_dir = Path("./LoRA_results_geometry")
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
    film_logs = defaultdict(lambda: {"train": [], "val": []})
    # boucle d'entrainement
    for epoch in range(epochs):
        model.unet.train()
        train_loss = 0.0
        film_train_sum = defaultdict(float)
        film_train_count = defaultdict(int)
        for inp, tar, caption, film_ids, sample_ids in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"): # affichage d'une barre de progression
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
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_loss = loss.item()
            train_loss += batch_loss
            for fid in film_ids:
                fid = int(fid)
                film_train_sum[fid] += batch_loss
                film_train_count[fid] += 1
        train_loss /= len(train_loader)
        train_loss_list.append(train_loss)
        for fid in film_train_sum:
            film_logs[fid]["train"].append(film_train_sum[fid] / film_train_count[fid])
        # validation
        # on évalue la perte sur le dataset de validation
        model.unet.eval()
        val_loss = 0.0
        film_val_sum = defaultdict(float)
        film_val_count = defaultdict(int)
        with torch.no_grad():
            for inp, tar, caption, film_ids, sample_ids in val_loader:
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
                    with torch.no_grad():
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
                batch_loss = loss.item()
                val_loss += batch_loss
                for fid in film_ids:
                    fid = int(fid)
                    film_val_sum[fid] += batch_loss
                    film_val_count[fid] += 1
        val_loss /= len(val_loader)
        val_loss_list.append(val_loss)
        for fid in film_val_sum:
            film_logs[fid]["val"].append(film_val_sum[fid] / film_val_count[fid])
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
        #for id in film_logs:
        #    film_logs[id]["train"][-1] = np.mean(film_logs[id]["train"])
        #    film_logs[id]["val"][-1]   = np.mean(film_logs[id]["val"])
        print(f"Epoch {epoch+1} | Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f}")
    return train_loss_list, val_loss_list, best_val_loss, film_logs     



# //////////////////////////////////////////////////
# 5- définition de la fonction de Grid Search LoRA
# //////////////////////////////////////////////////

########################################################################################
# fonction pour tracer les courbes d'apprentissage pour chacun des films de simulation
#########################################################################################

"""
film_logs ressemble à :
film_logs = {
    33: {"train": [], "val": []},
    40: {"train": [], "val": []},
    ...
}

"""

def plot_film_curves(film_logs, save_dir, exp_name):
    for film_id, logs in film_logs.items():
        film_path_dir = os.path.join(save_dir, f"film_{film_id}")
        os.makedirs(film_path_dir, exist_ok=True)
        plt.figure()
        plt.plot(logs["train"], label="Train")
        plt.plot(logs["val"], label="Val", linestyle="--")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.title(f"{exp_name} – Film {film_id}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(film_path_dir, f"{exp_name}_learning_curve.png"))
        plt.close()


#######################################################################################################
# boucle d'entrainement du modèle pré-entrainé et fine-tuné sur plusieurs films de simulation
#######################################################################################################



def run_experiment(model, train_loader, val_loader, 
                   save_dir, 
                   patience=3, epochs=100, lr=1e-4, device=DEVICE):
    os.makedirs(save_dir, exist_ok=True)
    # grille des configs LoRA
    r_values = [4]
    lora_alpha_values = [16]
    dropout_values = [0.2]
    batch_size = [4]
    epochs = [30]
    lr_values = [1e-5]
    use_prompt = [True]
    configs = list(product(r_values,
                    lora_alpha_values,
                    dropout_values,
                    batch_size,
                    epochs,
                    lr_values))
    results = []
    for (r, alpha, dropout, batch_size, epochs, lr_val) in configs:
        print(f"Training with LoRA config: r={r}, alpha={alpha}, dropout={dropout}, epochs={epochs}, lr={lr_val}")
        exp_name = f"r{r}_alpha{alpha}_dropout{dropout}_bs{batch_size}_ep{epochs}"
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
        train_loss_list, val_loss_list, best_val_loss, film_logs = train_lora(model_copy, 
                                                            train_loader, 
                                                            val_loader,
                                                            lora_config=lora_config, 
                                                            epochs=epochs, 
                                                            lr=lr_val, 
                                                            patience=patience, 
                                                            device=device)
        # enregistrement des resultas
        results.append({
            "config": exp_name,
            "best_val_loss": best_val_loss,
            "val_loss_history": val_loss_list,
            "train_loss_history": train_loss_list
        })
        plot_film_curves(film_logs, exp_dir, exp_name)
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
