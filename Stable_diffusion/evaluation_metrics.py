import torch
from pathlib import Path
from diffusers import StableDiffusionImg2ImgPipeline
from torchvision import transforms
import os
import matplotlib.pyplot as plt
from peft import PeftModel
import numpy as np

import stable_diffusion_multifilms
test_loader = stable_diffusion_multifilms.test_loader
train_loader = stable_diffusion_multifilms.train_loader
val_loader = stable_diffusion_multifilms.val_loader



# ---------------------------------
# loading du model
# ---------------------------------


model_path = Path("./git/ImageMLProject/Stable_diffusion/SD1-5_local")

device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    safety_checker=None  # IMPORTANT
).to(device)


# loader l'adaptateur LoRA dans le UNet du modèle

lora_model = Path("./LoRA_results_tri")
#model.unet.load_attn_procs(lora_model)

pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_model)
pipe.unet.to(device)

print("Modèle chargé avec LoRA dans l'UNet")



# --------------------------------------------
# récuperation d'un batch d'images
# --------------------------------------------

inp, tar, caption, film_ids, idx = next(iter(test_loader))


# conversion Tensor -> PIL image

to_pil = transforms.ToPILImage()
sample_image_pil = to_pil(inp[0]) 
tar_img = to_pil(tar[0]) 
frame = idx



# -----------------------------------------
# generation des predictions
# -----------------------------------------


save_dir = "./test_lora_tri"
os.makedirs(save_dir, exist_ok=True)

prompt = "CFD simulation of a 2D velocity field at timestep 1, fluid flow visualization, scientific data, heatmap style, spatial regression target"
prompt_none = ""


with torch.no_grad():
    prediction = pipe(
        prompt="",
        image=sample_image_pil,
        strength=0.0, # 0.75 par défaut
        guidance_scale=1.0, # 7.5 par défaut
        num_inference_steps=20 # 50 par défaut
    ).images[0] # PIL image


prediction.save(os.path.join(save_dir, "generated_image.png"))




# --------------------------------------------
# visualisation des champs physiques prédits
# --------------------------------------------


def cal_mae (pred_img, tar_img, title="", quantile_affichage=0.99):
    """
    _Summary_:
    _Args_:
    _Returns_:

    """
    pred_img = pred_img.convert("L")
    tar_img = tar_img.convert("L")
    # conversion des images en np.ndarray float32
    if isinstance(pred_img, torch.Tensor):
        pred_np = pred_img.permute(1,2,0).cpu().numpy()
    else:
        pred_np = np.array(pred_img).astype(np.float32) #[512,512,3] [0,1]
    if isinstance(tar_img, torch.Tensor):
        tar_np = tar_img.permute(1,2,0).cpu().numpy()
    else:
        tar_np = np.array(tar_img).astype(np.float32)
    # calcul de la mae
    mae_map = np.abs(pred_np - tar_np) # [H,W]
    mae_rel=100 * np.abs(pred_np - tar_np)
    mae_rel[np.nonzero(tar_np)] /= tar_np[np.nonzero(tar_np)]
    mae_rel[np.where(tar_np == 0)] = -1
    mae_max=np.quantile(mae_map, q=quantile_affichage)
    rel_max=np.quantile(mae_rel, q=quantile_affichage)
    fig, ax = plt.subplots(1, 4, sharex=True, sharey=True, figsize=(18,6))    
    im0 = ax[0].imshow(tar_np, vmin=0, vmax=tar_np.max(), cmap="viridis")
    ax[0].set_title("Target")
    ax[0].set_axis_off()
    im1 = ax[1].imshow(pred_np, vmin=0, vmax=tar_np.max(), cmap="viridis")
    ax[1].set_title("Prediction")
    ax[1].set_axis_off()
    im4 = ax[2].imshow(mae_map, cmap="viridis", vmin=0, vmax=mae_max)
    ax[2].set_title("MAE absolue")
    ax[2].set_axis_off()
    im5 = ax[3].imshow(mae_rel, cmap="coolwarm", vmin=0, vmax=rel_max)
    ax[3].set_title("MAE relative")
    ax[3].set_axis_off()
    fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04, label="m/s")
    fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04, label="m/s")
    fig.colorbar(im4, ax=ax[2], fraction=0.046, pad=0.04, label= "m/s" )
    fig.colorbar(im5, ax=ax[3], fraction=0.046, pad=0.04, label="%")
    fig.tight_layout()
    fig.suptitle(title, y=0.95)
    plt.savefig(os.path.join(save_dir, f"diff_map_{title}.png"))
    return mae_map


film_id = film_ids.item()
frame_idx = idx.item()

cal_mae(prediction, tar_img, title=f"Film_{film_id}_prediction{frame_idx}")


# ------------------------------------------------------------------
# calcul des mae moyennes des predictions pour l'ensemble du film
# ------------------------------------------------------------------

# récuperation des bornes des vitesses
import json


CFD_path = Path("./git/ImageMLProject/Datasets/CFD_Dataset/")
with open(os.path.join(CFD_path, f"data_film_tri_{film_id}_512x512.json"),"r") as f:
    bornes = json.load(f)

v_min = bornes["v_min"]
v_max = bornes["v_max"]




def evaluate_dataloader(pipe, 
                        dataloader, 
                        v_min, v_max):
    results = []
    with torch.no_grad():
        for inp, tar, caption, film_id, frame_idx in dataloader:
            inp_pil = to_pil(inp[0])
            tar_pil = to_pil(tar[0])
            pred_pil = pipe(
                prompt="",
                image=inp_pil,
                strength=0.0,
                guidance_scale=1.0,
                num_inference_steps=20
            ).images[0]
            # conversion grayscale
            pred_np = np.array(pred_pil.convert("L")).astype(np.float32)
            tar_np  = np.array(tar_pil.convert("L")).astype(np.float32) # (512,512) [0,255]
            # conversion en vitesse physique
            pred_vitesse = pred_np/255.0*(v_max-v_min)+v_min
            tar_vitesse = tar_np/255.0*(v_max-v_min)+v_min
            # masque géométrie / sol
            mask_valid = tar_vitesse > v_min + 1e-6
            # MAE physique
            err = np.abs(pred_vitesse - tar_vitesse)
            mae_mean = np.mean(err[mask_valid])
            results.append({
                "film_id": int(film_id.item()),
                "frame_idx": int(frame_idx.item()),
                "mae": mae_mean
            })
    return results


results_train = evaluate_dataloader(pipe, train_loader, 
                                    v_min=v_min, v_max=v_max)
results_val = evaluate_dataloader(pipe, val_loader, 
                                    v_min=v_min, v_max=v_max)
results_test = evaluate_dataloader(pipe, test_loader, 
                                    v_min=v_min, v_max=v_max)


# ---------------------------------
# concatenation des résultats
# ---------------------------------

for r in results_train:
    r["split"] = "train"



for r in results_val:
    r["split"] = "val"



for r in results_test:
    r["split"] = "test"



results_all = results_train + results_val + results_test



# -------------------------------
# tracé de la courbe
# -------------------------------



def plot_mae_per_film(results, film_ids, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    # --- filtrer le film ---
    colors = {"train": "green", "val": "blue", "test": "red"}
    plt.figure(figsize=(14, 6))
    for split in ["train", "val", "test"]:
        data = [r for r in results if r["split"] == split and r["film_id"] == film_id]
        if not data:
            continue
        data = sorted(data, key=lambda x: x["frame_idx"])
        frames = np.array([d["frame_idx"] for d in data])
        maes   = np.array([d["mae"] for d in data])
        # --- scatter ---
        plt.scatter(
            frames,
            maes,
            color=colors[split],
            label=f"{split} (frames)",
            alpha=0.7,
            s=25
        )
        # --- stats ---
        mean = np.mean(maes)
        std  = np.std(maes)
        # --- ligne moyenne ---
        plt.axhline(
            mean,
            color=colors[split],
            linestyle="--",
            linewidth=2,
            label=f"{split} mean = {mean:.3f} ± {std:.3f} m/s"
        )
        # --- bande écart-type ---
        plt.fill_between(
            [frames.min(), frames.max()],
            mean - std,
            mean + std,
            color=colors[split],
            alpha=0.15
        )
    plt.xlabel("Index temporel (frame du film)")
    plt.ylabel("MAE moyenne par image (m/s)")
    plt.title(f"MAE par frame – Film {film_id}")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"mae_film_{film_ids}.png")
    plt.savefig(save_path)
    plt.close()



plot_mae_per_film(
    results=results_all,
    film_ids=film_id,
    output_dir="./figures_mae_tri"
)
