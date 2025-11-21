
"""
Script qui teste le modèle stable_diffusion avec UNet entrainé avec LoRA

- loading du modèle pré-entrainé Stable_diffusion v1-5 et application de LoRA à l'UNet
- loading du dataset de test
- génération d'images à partir d'images d'entrée du dataset

"""

# 1- Import des librairies


import torch
from pathlib import Path
from diffusers import StableDiffusionImg2ImgPipeline
from torchvision import transforms
import os
import matplotlib.pyplot as plt
from peft import PeftModel, PeftConfig
import stable_main_LoRA_gridsearch
test_loader = stable_main_LoRA_gridsearch.test_loader



# 2- loading du modèle pré-entrainé avec LoRA

model_path = Path(
    "./git/ImageMLProject/Stable_diffusion/SD1-5_local"
)
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_path, 
    dtype=torch.float16).to("cuda")
device = "cuda" if torch.cuda.is_available() else "cpu"
# loader l'adaptateur LoRA dans le UNet du modèle

lora_model = Path("./LoRA_results")
#model.unet.load_attn_procs(lora_model)

pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_model)
pipe.unet.to(device)

print("Modèle chargé avec LoRA dans l'UNet")


# 3- Loading du dataset de test et d'une image d'entrée

# on récupère un batch d'images
inp, tar, caption = next(iter(test_loader))
print(f"Input batch shape: {inp.shape}, Inout type: {inp.dtype}")
# torch.Size([1, 3, 512, 512]) torch.float32 [0,1]
# np.min(inp.numpy()), np.max(inp.numpy()) # (0.0, 1.0)



# 4- génération d'images à partir d'images d'entrée du dataset


# on définit output_dir
save_dir = "./test_lora"
os.makedirs(save_dir, exist_ok=True)



# on convertit ce tensor en PIL image
to_pil = transforms.ToPILImage()
sample_image_pil = to_pil(inp[0]) # on prend la première image du batch
print(f"Sample image PIL size: {sample_image_pil.size}, mode: {sample_image_pil.mode}")
# Sample image PIL size: (512, 512), mode: RGB

# génération de la prediction
prompt = "CFD simulation of a 2D velocity field at timestep 1, fluid flow visualization, scientific data, heatmap style, spatial regression target"
prompt_none = ""

with torch.no_grad():
    prediction = pipe(
        prompt=prompt,
        image=sample_image_pil,
        strength=0.0000001, # 0.75 par défaut
        guidance_scale=1.0, # 7.5 par défaut
        num_inference_steps=30, # 50 par défaut
    ).images[0] # PIL image

print(f"Prediction size: {prediction.size}, mode: {prediction.mode}")
# Prediction size: (512, 512), mode: RGB


# sauvegarde de la prediction
prediction.save(os.path.join(save_dir, "generated_image.png"))
# visualisation de la prediction
plt.imshow(prediction)
plt.axis("off")
plt.show()




# visualisation des resultats avec l'image target

tar_img = to_pil(tar[0]) # target de la première image du batch
tar_img.save(os.path.join(save_dir, "target_image.png"))


fig, ax = plt.subplots(1, 3, figsize=(18, 6))
ax[0].imshow(sample_image_pil)
ax[0].set_title("Input Image")
ax[0].axis("off")
ax[1].imshow(prediction)
ax[1].set_title("Generated Image")
ax[1].axis("off")
ax[2].imshow(tar_img)
ax[2].set_title("Target Image")
ax[2].axis("off")

plt.savefig(os.path.join(save_dir, "comparison_with_target.png"))
plt.show()




# //////////////////////////////////////
# test des paramètres 
# //////////////////////////////////////
from itertools import product
import time
save_dir = "./Lora_results_gridsearch"
os.makedirs(save_dir, exist_ok=True)
# grille des paramètres
strength_values = [0.0, 0.1, 0.3]
guidance_values = [1.0, 5.0, 7.0]
step_values = [10, 20]
prompt_value = ["CFD simulation of a 2D velocity field at timestep 1, fluid flow visualization, scientific data, heatmap style, spatial regression target",
                ""]
results = {}
to_pil = transforms.ToPILImage()
tar_img = to_pil(tar[0])
for strength, guidance_scale, steps, prompt_val in product(strength_values, guidance_values, step_values, prompt_value):
    print(f"\n➡ Running inference with "
          f"strength={strength}, guidance={guidance_scale}, steps={steps}, prompt={prompt_val}")
    with torch.no_grad():
        prediction = pipe(
            prompt=prompt_val,
            image=sample_image_pil,
            strength=strength, # 0.75 par défaut
            guidance_scale=guidance_scale, # 7.5 par défaut
            num_inference_steps=steps, # 50 par défaut
        ).images[0] # PIL image
    key = f"S{strength}_G{guidance_scale}_N{steps}"
    prediction.save(os.path.join(save_dir, f"{key}.png"))
    results[key]=prediction


# /////////////////////////////////////////
# figure des images
# /////////////////////////////////////////

import math

keys = sorted(results.keys())
num_images = len(keys)
cols = 12
rows = math.ceil(num_images / cols)
plt.figure(figsize=(4*cols, 4*rows))
for i, key in enumerate(keys):
    plt.subplot(rows, cols, i+1)
    plt.imshow(results[key])
    plt.axis("off")
    plt.title(key, fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "fig_totale.png"))
plt.show()

# 5- Calcul de la MAE

# récupération des tensors
tar_tensor = tar[0] # torch.Size([3, 512, 512]) torch.float32 [0,1]
pred_tensor = transforms.ToTensor()(prediction) # torch.Size([3, 512, 512]) torch.float32 [0,1]
