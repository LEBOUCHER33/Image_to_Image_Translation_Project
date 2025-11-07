from pathlib import Path
import os
import numpy as np
from CFD_dataset_cas_2 import load_dataset, get_tf_dataset, resize_and_normalize
import json
import pandas as pd
import tensorflow as tf
import keras
import matplotlib.pyplot as plt


"""
script qui test la performance du meilleur modèle entrainé 

stratégie d'entrainement :

    - dataset train = 720 couples d'images (input, taget) du film 3 mélangées random
    - dataset val = 180 couples d'images (input, taget) du film 3 mélangées random
    - dataset test = 90 couples d'images (input, taget) du film 3 mélangées random

Workflow :

1- récupération du modèle ayant la loss la plus faible
2- loading G, D et best_criteria à partir du checkpoint enregistré par Ray
3- génération des prédictions du dataset de test
4- calcule de la MAE sur les vitesses sur les predictions individuelles
5- calcule des MAE globales sur les datasets de train/val/test
6- affichage de la courbe des erreurs
7- affichage des predictions et du mapping des erreurs


"""


DEBUG = False

# 1- Datasets

dataset_name = "film_3_128x128.npy"
dataset_path = os.path.abspath("./git/ImageMLProject/Datasets/CFD_Dataset/" + dataset_name)
dataset_path = Path(dataset_path)
dataset_path = dataset_path.resolve()

inputs, targets, train_idx, test_idx, val_idx = load_dataset(dataset_path, offset=1)
x_train, y_train = resize_and_normalize(inputs[train_idx], targets[train_idx])
x_val, y_val = resize_and_normalize(inputs[val_idx], targets[val_idx])
x_test, y_test = resize_and_normalize(inputs[test_idx], targets[test_idx])
test_dataset = get_tf_dataset(x_test, y_test, BATCH_SIZE=1, shuffle=False)
train_dataset = get_tf_dataset(x_train, y_train, BATCH_SIZE=1, shuffle=False)
val_dataset = get_tf_dataset(x_val, y_val, BATCH_SIZE=1, shuffle=False)

if DEBUG:
    print("shape :", x_train.shape, y_train.shape)
    print("min :", np.min(x_train), np.min(y_train))
    print("max :", np.max(x_train), np.max(y_train))
    print("type :", type(x_train[0]), type(y_train[0]))
    


# 2- variables des bornes du film

CFD_path = Path("./git/ImageMLProject/Datasets/CFD_Dataset/")
with open(os.path.join(CFD_path, "data_film_3_128x128.json"),"r") as f:
    bornes = json.load(f)

v_min = bornes["v_min"]
v_max = bornes["v_max"]





# 3- identification du meilleur checkpoint

ray_results_dir = os.path.abspath(
    "./my_ray_results_cas_8/cas_8_run_1")
os.makedirs(ray_results_dir, exist_ok=True)
print(ray_results_dir)

dataframes = []

for trial_dir in os.listdir(ray_results_dir):
    trial_path = os.path.join(ray_results_dir, trial_dir)
    if os.path.isdir(trial_path):
        results_file = os.path.join(trial_path, "result.json")
        params_file = os.path.join(trial_path, "params.json")
        if os.path.isfile(params_file) and os.path.isfile(results_file):
            # Vérifie si les fichiers sont vides
            if os.path.getsize(params_file) == 0:
                print(f"Fichier vide ignoré : {params_file}")
                continue
            if os.path.getsize(results_file) == 0:
                print(f"Fichier vide ignoré : {results_file}")
                continue
            try:
                with open(params_file, "r") as f:
                    params = json.load(f)
                # Lecture du fichier résultats directement via le chemin
                result_json = pd.read_json(results_file, lines=True)
                result_json["trial_id"] = trial_dir
                dataframes.append(result_json)
            except (json.JSONDecodeError, TypeError, ValueError) as e:
                print(f"Erreur de lecture dans {trial_dir} : {e}")
                continue


if not dataframes:
    raise ValueError("Aucun fichier de résultats valide trouvé.")


# je réunis les trials dans un dataframe
result_df = pd.concat(dataframes, ignore_index=True)
# je recupère uniquement la métrique la plus performante / trial
df_best = result_df.loc[result_df.groupby(
    "trial_id")["best_criteria"].idxmin()]
print(df_best.shape[0])
# je trie par ordre ascendant
df_best_sorted = df_best.sort_values("best_criteria")
# Identifie le meilleur entraînement
best_trial = df_best_sorted.iloc[0]
print("====== Meilleur entraînement ======")
print(f"Trial ID        : {best_trial['trial_id']}")
print(f"Best criteria   : {best_trial['best_criteria']}")
print(f"Training step   : {best_trial['training_iteration']}")
best_trial_id = best_trial['trial_id']
params_path = os.path.join(ray_results_dir, best_trial_id, "params.json")

if os.path.isfile(params_path):
    with open(params_path, "r") as f:
        best_params = json.load(f)
    print("------ Hyperparamètres ------")
    for k, v in best_params.items():
        print(f"{k}: {v}")


# checkpoint path
best_trial_path = os.path.join(ray_results_dir, best_trial_id)


checkpoint_dirs = [
    d for d in os.listdir(best_trial_path)
    if os.path.isdir(os.path.join(best_trial_path, d)) and d.startswith("checkpoint_")
]
if not checkpoint_dirs:
    raise FileNotFoundError("Aucun dossier de checkpoint trouvé.")



# Option 1 : prendre le dernier checkpoint (ordre alphanumérique)
checkpoint_dirs.sort()
latest_checkpoint = checkpoint_dirs[-1]
# Chemin complet du dernier checkpoint
checkpoint_path = os.path.join(best_trial_path, latest_checkpoint)
print("Chemin du checkpoint :", checkpoint_path)




# 4- loading du meilleur modèle entrainé

generator = keras.models.load_model(os.path.join(
    checkpoint_path, "generator.keras"))
discriminator = keras.models.load_model(os.path.join(
    checkpoint_path, "discriminator.keras"))
best_criteria_file = os.path.join(
    checkpoint_path, "training_state.json")



# 4- évaluation du modèle

output_dir = os.path.abspath("./predictions_pix2pix_CFD_cas_8")
os.makedirs(output_dir, exist_ok=True)



def prediction(initial_image, target_image, v_min, v_max):
    """
    _Summary_: fonction qui génère une prediction et calcule l'erreur sur la vitesse de prediction vs vitesse cible
    _Args_: 
        - initial_image (np.array): image d'entrée (shape (1,H,W,1))
        - target_image (np.array): image cible (shape (1,H,W,1))
        - v_min, v_max (float): bornes de la vitesse physique
    _Returns_: 
        - prediction (np.array) : image predite [0,255]
        - MAE de la vitesse (float)
    
    """
    # prediction
    current_pred = generator(initial_image, training=False)  # tenseur (b, C, W, H) [-1,1]
    pred = ((current_pred[0].numpy()+1) * 127.5) # np.darray (H,W,1), [0,255] /// on dénormalise
    # conversion en vitesse physique
    pred_vitesse = pred/255*(v_max-v_min)+v_min # [0,255] -> [0,1] -> [v_min,v_max]
    tar_vitesse = (target_image[0].numpy()+1)/2*(v_max-v_min)+v_min # [-1,1] -> [0,1] -> [v_min,v_max]
    # calculs d'erreurs
    diff_vitesse = pred_vitesse - tar_vitesse
    err_mae = np.abs(diff_vitesse) # np.ndarray (H,W,1) contenant l'erreur absolue pixel par pixel
    err_mse = np.square(diff_vitesse)
    # moyennes des erreurs et dispersion
    mae_mean = np.mean(err_mae) # erreur moyenne sur l'image
    mse_mean = np.mean(err_mse)
    rmse_mean = np.sqrt(mse_mean)
    var_mae = np.var(err_mae)
    return pred, mae_mean, mse_mean, rmse_mean, var_mae




if DEBUG:
    inp = x_test[0:1] 
    tar = y_test[0:1]
    print(inp.shape, tar.shape, type(inp), type(tar), np.min(inp), np.max(inp)) # tensors ([1,128,128,1]) [-1,1]
    tar_arr = ((tar.numpy()+1)*127.5) # np.ndarray (1,128,128,1) [0,255]
    tar_arr_3d = ((tar[0].numpy()+1)*127.5) # (128,128,1) [0,255] on supp la dimension batch tar[0]
    pred, loss_mae, loss_mse, loss_rmse, var_mae = prediction(inp, tar, v_min, v_max)
    print(pred.shape, type(pred), np.min(pred), np.max(pred)) # np.array (128,128,1) [0,255]
    print("MAE_loss :", loss_mae)
    # pour visualiser les images on ramène en 2D et on dénormalise
    image_inp = inp[0,:,:,0] # tensor ([128,128]) [-1,1]
    img_inp = ((image_inp.numpy()+1)*127.5) # numpy.ndarray [0,255]
    img_inp = img_inp.astype(np.uint8)
    plt.imshow(img_inp)
    plt.show()   
    # pour visualiser la prediction on doit supprimer la dernière = (H,W)
    image = pred[:,:,0].astype(np.uint8) # ou image = pred.squeeze().astype(np.uint8)
    plt.imshow(image)
    plt.show()



# on va calculer l'erreur individuelle sur chacune des predictions du dataset de test et la variance
error_maps = []
mae_list = []
rmse_list = []
for i in range(x_test.shape[0]):
    inp = x_test[i:i+1] # tensor ([1,128,128,1]) normalise
    tar = y_test[i:i+1]
    pred, test_mae, test_mse, test_rmse, var_mae = prediction(inp, tar, v_min, v_max)
    mae_list.append(test_mae)
    rmse_list.append(test_rmse)
    





# prediction adaptée à un dataset

def prediction_dataset(inputs, targets, v_min, v_max):
    """
    _Summary_: fonction qui génère des predictions sur un dataset et calcule les erreurs globales
    _Args_: 
        - inputs (np.array): images d'entrée (shape (N,H,W,1))
        - targets (np.array): images cibles (shape (N,H,W,1))
        - v_min, v_max (float): bornes de la vitesse physique
    _Returns_: 
        - preds (np.array) : images predite [0,255]
        - MAE de la vitesse (float)
        - RMSE de la vitesse (float)
        - liste des MAE individuelles (list)
        - liste des RMSE individuelles (list)
        - variance des MAE individuelles (float)
    
    """
    mae_list = []
    mse_list = []
    rmse_list = []
    var_list = []
    preds_list = []
    for i in range(inputs.shape[0]):
        inp = inputs[i:i+1] # tensor ([1,128,128,1]) normalise
        tar = targets[i:i+1]
        pred, mae_mean, mse_mean, rmse, var_mae = prediction(inp, tar, v_min, v_max)
        mae_list.append(mae_mean)
        mse_list.append(mse_mean)
        rmse_list.append(rmse)
        var_list.append(var_mae)
        preds_list.append(pred[np.newaxis, ...]) # oblige à garder la dimension batch [N,128,128,1]
    mae_dataset = np.mean(mae_list) # moyenne des MAE individuelles sur le dataset
    mse_dataset = np.mean(mse_list)
    rmse_dataset = np.sqrt(mse_dataset)
    var_dataset = np.var(mae_list) # variance des MAE individuelles sur le dataset
    preds = np.concatenate(preds_list, axis=0) 
    var_mae = np.var(mae_list) # variance des MAE individuelles sur le dataset
    return preds, mae_dataset, rmse_dataset, mae_list, rmse_list, var_dataset



# on veut calculer la valeur de la mae sur chacun des datasets
preds_test, mae_dataset_test, rmse_dataset_test, test_mae_list, test_rmse_list, test_var = prediction_dataset(x_test, y_test, v_min, v_max)
preds_train, mae_dataset_train, rmse_dataset_train, train_mae_list, train_rmse_list, train_var = prediction_dataset(x_train, y_train, v_min, v_max)
preds_val, mae_dataset_val, rmse_dataset_val, val_mae_list, val_rmse_list, val_var = prediction_dataset(x_val, y_val, v_min, v_max)

if DEBUG:
    print("MAE Test :", mae_dataset_test)
    print("MAE Train :", mae_dataset_train)
    print("MAE Val :", mae_dataset_val)
    print("Shape preds_test :", preds_test.shape)

# //////////////////////
# courbe des pertes
# //////////////////////
def plot_loss_curve(metric_name, values_list, dataset_means, output_dir):
    """
    Affiche et sauvegarde la courbe des pertes (MAE, MSE, RMSE) sur le dataset de test.

    Args:
        metric_name (str): Nom de la métrique ("MAE", "MSE" ou "RMSE").
        values_list (list): Liste des pertes individuelles par image.
        dataset_means (dict): Moyennes globales des pertes {"train": x, "val": y, "test": z}.
        output_dir (str): Dossier de sauvegarde.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(values_list, marker='o', markersize=3, linewidth=1, color='tab:blue', label=f'{metric_name} par image')

    plt.xlabel("Index image")
    plt.ylabel(f"Loss ({metric_name})")
    plt.title(f"Courbe des pertes {metric_name} sur le dataset de test")
    plt.grid(True, linestyle='--', alpha=0.5)

    # Lignes horizontales pour les moyennes globales
    plt.axhline(dataset_means['test'], color='red', linestyle='--', linewidth=2,
                label=f"{metric_name}_test: {dataset_means['test']:.4f}")
    plt.axhline(dataset_means['train'], color='green', linestyle='--', linewidth=2,
                label=f"{metric_name}_train: {dataset_means['train']:.4f}")
    plt.axhline(dataset_means['val'], color='blue', linestyle='--', linewidth=2,
                label=f"{metric_name}_val: {dataset_means['val']:.4f}")

    plt.legend(loc='upper right', fontsize=9)
    plt.tight_layout()

    filename = f"courbe_des_pertes_{metric_name.lower()}.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

# dict des métriques d'erreurs à afficher
metrics = {
    "MAE": {
        "values": mae_list,
        "means": {
            "train": mae_dataset_train,
            "val": mae_dataset_val,
            "test": mae_dataset_test
        }
    },
    "RMSE": {
        "values": rmse_list,
        "means": {
            "train": rmse_dataset_train,
            "val": rmse_dataset_val,
            "test": rmse_dataset_test
        }
    }
}

fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
for ax, (name, data) in zip(axes, metrics.items()):
    ax.plot(data["values"], marker='o', markersize=3, color='tab:blue', label=name)
    for split, color in zip(["train", "val", "test"], ["green", "blue", "red"]):
        ax.axhline(data["means"][split], color=color, linestyle='--', label=f"{split}: {data['means'][split]:.4f}")
    ax.set_ylabel(f"{name}")
    ax.grid(True, linestyle='--', alpha=0.5)
axes[-1].set_xlabel("Index image")
fig.suptitle("Courbes de pertes par métrique")
fig.tight_layout()
fig.legend(loc='upper right')
plt.savefig(os.path.join(output_dir, "courbe_des_pertes_metriques.png"), dpi=300)
plt.show()





# //////////////////////////////////////////////////////////////////////////////////////////
# courbe des pertes globales sur les trois datasets avec affichage de la variance globale
# /////////////////////////////////////////////////////////////////////////////////////////


all_data = []

# on regroupe les données
for idx, mae in zip (train_idx, train_mae_list):
    all_data.append((idx, mae, 'train'))

for idx, mae in zip (val_idx, val_mae_list):
    all_data.append((idx, mae, 'val'))

for idx, mae in zip (test_idx, test_mae_list):
    all_data.append((idx, mae, 'test'))    

# on trie selon l'index d'origine
all_data.sort(key=lambda x: x[0])

# tableaux numpy
indices = np.array([item[0] for item in all_data])
mae_values = np.array([item[1] for item in all_data])
labels = np.array([item[2] for item in all_data])   


# figure
plt.figure(figsize=(12, 6))
colors = {'train': 'green', 'val': 'blue', 'test': 'red'}
for split in ['train', 'val', 'test']:
    mask = labels == split
    plt.errorbar(indices[mask], 
                 mae_values[mask], 
                 fmt='o', markersize=4, 
                 label=split, 
                 color=colors[split], alpha=0.7)

plt.axhline(mae_dataset_train, color='green', linestyle='--', label=f"MAE_train: {mae_dataset_train:.4f} m/s")
plt.fill_between(indices, mae_dataset_train - train_var, mae_dataset_train + train_var, color='green', alpha=0.2, label=f"Variance_train : {train_var:.4f} (m/s)²")
plt.axhline(mae_dataset_val, color='blue', linestyle='--', label=f"MAE_val: {mae_dataset_val:.4f} m/s")
plt.fill_between(indices, mae_dataset_val - val_var, mae_dataset_val + val_var, color='blue', alpha=0.2, label=f"Variance_val : {val_var:.4f} (m/s)²")
plt.axhline(mae_dataset_test, color='red', linestyle='--', label=f"MAE_test: {mae_dataset_test:.4f} m/s")
plt.fill_between(indices, mae_dataset_test - test_var, mae_dataset_test + test_var, color='red', alpha=0.2, label=f"Variance_test : {test_var:.4f} (m/s)²")
plt.xlabel("Index image dans le film")
plt.ylabel("MAE par image (moyenne des erreurs absolues sur la vitesse) m/s")
plt.title("Courbe des pertes MAE moyenne par prediction sur les trois datasets")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
#plt.savefig(os.path.join(output_dir, "courbe_des_pertes_mae_avec_variance.png"), dpi=300)
plt.show()    


# //////////////////////////
# box_plot des MAE
# //////////////////////////

import seaborn as sns


colors=sns.color_palette('coolwarm',3)
data=[train_mae_list, val_mae_list, test_mae_list]
labels=['train', 'val', 'test']
plt.figure(figsize=(10, 8))
box = plt.boxplot(
    data,                   # argument positionnel obligatoire
    vert=False,             # boxplots horizontaux
    patch_artist=True,      # pour pouvoir colorer les boîtes
    labels=labels,
    showmeans=True,
    meanline=True,
    meanprops={'color': 'black', 'linewidth': 2},
    medianprops={'color': 'black', 'linewidth': 1.5},
    flierprops={'marker': 'o', 'markerfacecolor': 'blue', 'markersize': 5}
)
# Appliquer les couleurs
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

mean_line = plt.Line2D([], [], color='black', linewidth=2, label='Mean')
median_line = plt.Line2D([], [], color='black', linestyle='--', linewidth=2, label='Median')
# Mise en forme
plt.xlabel("MAE moyenne par image prédite [m/s]")
plt.ylabel("Dataset")
plt.title("Dispersion des erreurs absolues moyennes par prédiction sur les trois datasets", fontsize=13)
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(handles=[mean_line, median_line], loc='upper right')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "boxplot_des_pertes_mae.png"), dpi=300, bbox_inches='tight')
plt.show()



# //////////////////////////////////
# courbe des RMSE moyens sur les trois datasets
# //////////////////////////////////

all_data_rmse = []

# on regroupe les données
for idx, rmse in zip (train_idx, train_rmse_list):
    all_data_rmse.append((idx, rmse, 'train'))

for idx, rmse in zip (val_idx, val_rmse_list):
    all_data_rmse.append((idx, rmse, 'val'))

for idx, rmse in zip (test_idx, test_rmse_list):
    all_data_rmse.append((idx, rmse, 'test'))    

# on trie selon l'index d'origine
all_data_rmse.sort(key=lambda x: x[0])

# tableaux numpy
indices = np.array([item[0] for item in all_data_rmse])
rmse_values = np.array([item[1] for item in all_data_rmse])
labels = np.array([item[2] for item in all_data_rmse])   

# figure
plt.figure(figsize=(12, 6))
colors = {'train': 'green', 'val': 'blue', 'test': 'red'}
for split in ['train', 'val', 'test']:
    mask = labels == split
    plt.errorbar(indices[mask], 
                 rmse_values[mask], 
                 fmt='o', markersize=4, 
                 label=split, 
                 color=colors[split], alpha=0.7)

plt.axhline(rmse_dataset_train, color='green', linestyle='--', label=f"RMSE_train: {rmse_dataset_train:.4f} m/s")
plt.axhline(rmse_dataset_val, color='blue', linestyle='--', label=f"RMSE_val: {rmse_dataset_val:.4f} m/s")
plt.axhline(rmse_dataset_test, color='red', linestyle='--', label=f"RMSE_test: {rmse_dataset_test:.4f} m/s")
plt.xlabel("Index image dans le film")
plt.ylabel("RMSE par image (moyenne des erreurs quadratiques sur la vitesse) m/s")
plt.title("Courbe des pertes RMSE moyenne par prediction sur les trois datasets")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "courbe_des_pertes_rmse.png"), dpi=300)
plt.show()    




# ////////////////////////////////////////////////////////////////////////                                                      
# affichage des images / des erreurs absolues / des erreurs quadratiques
# ////////////////////////////////////////////////////////////////////////
from scipy import stats


for i in range (3, x_test.shape[0], 15):
    # Données d'entrée et cible
    input_i = x_test[i]
    tar_i = y_test[i]
    pred_i = preds_test[i]
    # Conversion en images [0,255]
    img_inp = np.squeeze((input_i.numpy() + 1) * 127.5).astype(np.uint8)  # (128,128)
    img_tar = np.squeeze((tar_i.numpy() + 1) * 127.5).astype(np.uint8)
    img_pred = np.squeeze(pred_i).astype(np.uint8)
    # Conversion en vitesses physiques
    input_vitesse = np.squeeze((input_i.numpy() + 1) / 2) * (v_max - v_min) + v_min # [-1,1] -> [0,1] -> [v_min,v_max]
    pred_vitesse = np.squeeze(pred_i) / 255 * (v_max - v_min) + v_min # [0,255] -> [0,1] -> [v_min,v_max]
    tar_vitesse = np.squeeze((tar_i.numpy() + 1) / 2) * (v_max - v_min) + v_min # [-1,1] -> [0,1] -> [v_min,v_max]
   # Carte d’erreur mae : erreur locale absolue
    error_map_mae_rel = np.abs(pred_vitesse - tar_vitesse) / (np.maximum(0, np.abs(tar_vitesse)))
    error_map_mae_abs = np.abs(pred_vitesse - tar_vitesse)
    error_rel_max = np.max(error_map_mae_rel)
    error_max = np.max(error_map_mae_abs)
    error_min = np.min(error_map_mae_abs)
    error_variance = np.var(error_map_mae_abs)
    # Carte d’erreur mse : erreur locale quadratique
    error_map_quad = np.square(pred_vitesse - tar_vitesse)
    mae_mean = np.mean(error_map_mae_abs)
    mse_mean = np.mean(error_map_quad)
    rmse_mean = np.sqrt(mse_mean)
    error_map_rmse = np.sqrt(error_map_quad)
    # Carte d'affichage des gradients temporels des vitesses
    grad_mag = np.abs(tar_vitesse - input_vitesse)
    # compression 1D
    grad_mag_1d = grad_mag.flatten()
    error_map_mae_abs_1d = error_map_mae_abs.flatten()
    mask = np.isfinite(error_map_mae_abs_1d) & np.isfinite(grad_mag_1d)
    x=grad_mag_1d[mask]
    y=error_map_mae_abs_1d[mask]
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    # affichage
    """
    On veut une échelle comparative entre les images et avec l'error_map
        - pour l'affichage des images je prends l'échelle 0/v_max du film
        - pour l'error_map_abs je prends l'échelle 0/max(error_map_mae_abs)
        - pourm'erreur mae relative je prends l'échelle 0/10 (on affiche les erreurs relatives inférieures à 10x la valeur réelle)
        - pour l'error_map_quad je prends l'échelle 0/max(error_map_quad.max)
    """
    fig, ax = plt.subplots (1,4,sharey=True,figsize=(15,8))
    # Échelle commune pour les vitesses
    vmin_img, vmax_img = 0, v_max
    # Images de vitesse
    im0 = ax[0].imshow(img_tar, vmin=0, vmax=vmax_img)
    ax[0].set_title("Target")
    ax[0].set_axis_off()
    im1 = ax[1].imshow(img_pred, vmin=0, vmax=vmax_img)
    ax[1].set_title("Prediction")
    ax[1].set_axis_off()
    im2 = ax[2].imshow(error_map_mae_abs, vmin=0, vmax=error_max)
    ax[2].set_title("MAE prediction")
    ax[2].set_axis_off()
    im3 = ax[3].imshow(error_map_rmse, vmin=0, vmax=10)
    ax[3].set_title("RMSE prediction")
    ax[3].set_axis_off()
    fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04, label="m/s")
    fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04, label="m/s")
    fig.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04, label="m/s")
    fig.colorbar(im3, ax=ax[3], fraction=0.046, pad=0.04, label="m/s")
    # ---- ANNOTATIONS ----
    stats_text = (
        f"erreur_abs_max = {error_max:.3f} m/s\n"
        f"erreur_abs_min = {error_min:.3f} m/s\n"
        f"erreur_abs_variance = {error_variance:.3f} (m/s)²\n"
        f"MAE moyenne = {mae_mean:.3f} m/s\n"
        f"MSE moyenne = {mse_mean:.3f} (m/s)²\n"
        f"RMSE moyenne = {rmse_mean:.3f} m/s"
    )
    fig.suptitle(f"Prediction {i} — Erreurs absolues sur la prédiction de vitesse", fontsize=16, y=0.95, color='black')
    fig.text(0.01, 0.85, stats_text, fontsize=12, color='black', va='top')
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, f'fig_pred_mae_abs_{i}.png'))
    # Cartes d'erreur
    fig, ax = plt.subplots(1, 4, sharex=True, sharey=True, figsize=(18,8))    
    im0 = ax[0].imshow(img_tar, vmin=0, vmax=vmax_img)
    ax[0].set_title("Target")
    ax[0].set_axis_off()
    im1 = ax[1].imshow(img_pred, vmin=0, vmax=vmax_img)
    ax[1].set_title("Prediction")
    ax[1].set_axis_off()
    im4 = ax[2].imshow(error_map_mae_rel, cmap="coolwarm", vmin=0, vmax=error_rel_max)
    ax[2].set_title("MAE relative")
    ax[2].set_axis_off()
    im5 = ax[3].imshow(grad_mag, cmap="coolwarm", vmin=0)
    ax[3].set_title("Gradient de la vitesse")
    ax[3].set_axis_off()
    fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04, label="m/s")
    fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04, label="m/s")
    fig.colorbar(im4, ax=ax[2], fraction=0.046, pad=0.04, label= "mae relative" )
    fig.colorbar(im5, ax=ax[3], fraction=0.046, pad=0.04, label="m²/s²")
        # ---- ANNOTATIONS ----
    # Ajoute les statistiques dans la figure
    stats_text = (
        f"MAE = {mae_mean:.3f} m/s\n"
        f"MSE = {mse_mean:.3f} (m/s)²\n"
        f"RMSE = {rmse_mean:.3f} m/s"
    )
    fig.suptitle(f"Prediction {i} — Erreurs sur la prédiction de vitesse", fontsize=16, y=0.95, color='black')
    fig.text(0.01, 0.85, stats_text, fontsize=12, color='black', va='top')
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, f'fig_pred_mae-gradient_{i}.png'))
    # subplot quantification de l'erreur
    fig, ax = plt.subplots(1,4, figsize=(15,4), gridspec_kw={'width_ratios': [1, 1, 1, 1]})
    im0=ax[0].imshow(error_map_mae_rel, cmap="coolwarm", vmin=0, vmax=error_rel_max)
    ax[0].set_title("MAE relative")
    ax[0].set_axis_off()
    fig.colorbar(im0, ax=ax[0],fraction=0.046, pad=0.04, label="mae relative")
    im1=ax[1].imshow(grad_mag, cmap="coolwarm", vmin=0)
    ax[1].set_title("Gradient de la vitesse")
    ax[1].set_axis_off()
    fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04, label="m/s/pixel")
    # histogramme
    counts, bins, patches = ax[3].hist(error_map_mae_abs.flatten(), bins=np.arange(0, error_max.max() + 1, 1),
                               color='darkblue', alpha=0.7, edgecolor='black')
    count_0 = counts[0]
    count_0_ratio = (count_0 / counts.sum()) * 100
    ax[3].annotate(f"{int(count_0)} pixels\n {count_0_ratio:.2f}%",
                 xy=(bins[0], count_0),
                 xycoords='data',
                 xytext=(25,0.2),
                 textcoords='offset points',
                 color='darkblue',
                 fontsize=8,
                 bbox=dict(facecolor='white', alpha=0.8)
    )
    seuil = 9.2
    pixels_sous_seuil = np.sum(error_map_mae_abs < seuil)
    pixels_total = error_map_mae_abs.size
    pourcentage_pixels_sous_seuil = (pixels_sous_seuil / pixels_total) * 100
    fig.text(0.5,0.5,f"Pixels avec MAE < 5% d'erreur absolue: \n {pixels_sous_seuil} / {pourcentage_pixels_sous_seuil:.2f}%",
                 fontsize=8, color='darkblue',
                 transform=plt.gca().transAxes,
                 ha='center', va='center',bbox=dict(facecolor='white', alpha=0.8))
    ax[3].set_xlabel("Erreur absolue (m/s)")
    ax[3].set_ylabel("Fréquence en nombre de pixels")
    ax[3].set_xticks(np.arange(0, int(error_max.max()) + 1, 1))
    ax[3].set_xlim(0, 10)
    ax[3].set_ylim(0, counts.max()*1.1)
    ax[3].set_title("Distribution des erreurs absolues")
    x = grad_mag_1d[np.isfinite(error_map_mae_rel.flatten())]
    y = error_map_mae_rel.flatten()[np.isfinite(error_map_mae_rel.flatten())]
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    ax[2].scatter(x, y, 
                  color='darkblue', alpha=0.5, s=5)
    ax[2].plot(x, slope * x + intercept, color='red', linestyle='--',
               )
    ax[2].set_xlabel("Gradient de la vitesse (m/s/pixel)")
    ax[2].set_ylabel("Erreur relative (%)")
    ax[2].set_title("Erreurs relatives vs gradients")
    coeff_corr = np.corrcoef(grad_mag.flatten(), error_map_mae_abs.flatten())[0,1]
    ax[2].text(0.05, 0.95, f"r = {coeff_corr:.3f}", transform=ax[2].transAxes,
           fontsize=12, color='red', ha='left', va='top')
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, f'hist_pred_mae_hist_{i}.png'))
    plt.show()
    plt.close()




# /////////////////////////////////////////////////////////////////////
# calcul de la correlation erreurs absolues vs gradients sur le film
# /////////////////////////////////////////////////////////////////////

grad_list, error_list, error_rel_list = [], [], []

for i in range(x_test.shape[0]):
    input_i = x_test[i]
    tar_i = y_test[i]
    pred_i = preds_test[i]
    input_vitesse = np.squeeze((input_i.numpy() + 1) / 2) * (v_max - v_min)
    tar_vitesse = np.squeeze((tar_i.numpy() + 1) / 2) * (v_max - v_min)
    pred_vitesse = np.squeeze(pred_i) / 255 * (v_max - v_min)
    grad_mag = np.abs(tar_vitesse - input_vitesse)
    error_map_mae_abs = np.abs(pred_vitesse - tar_vitesse)
    error_map_rel = np.abs(pred_vitesse - tar_vitesse) / (np.abs(tar_vitesse))
    error_map_rel*=100
    error_rel_list.append(error_map_rel.flatten())
    grad_list.append(grad_mag.flatten())
    error_list.append(error_map_mae_abs.flatten())


grad_all = np.concatenate(grad_list)
error_all = np.concatenate(error_list)
err_rel_all = np.concatenate(error_rel_list)
# --- Suppression des NaN / Inf ---
mask = np.isfinite(grad_all) & np.isfinite(error_all) & np.isfinite(err_rel_all)
grad_all = grad_all[mask]
error_all = error_all[mask]
err_rel_all = err_rel_all[mask]


# --- Correlation linéaire pour MAE ---
coeff_corr = np.corrcoef(grad_all, error_all)[0,1]
slope, intercept, r_value, p_value, std_err = stats.linregress(grad_all, error_all)


# --- Correlation linéaire pour relative errors ---
coeff_corr_rel = np.corrcoef(grad_all, err_rel_all)[0,1]
s, i, r, p, std = stats.linregress(grad_all, err_rel_all)


if DEBUG:
    print(grad_all.shape, error_all.shape, coeff_corr)

# //////////////////////////
# Scatter_plot
# //////////////////////////

fig, ax = plt.subplots(1,2,figsize=(10,6))
ax[0].scatter(grad_all, error_all, alpha=0.5, s=2, color='royalblue', label='Predictions')
ax[0].plot(grad_all, slope * grad_all + intercept, color='red', linestyle='--')
ax[0].text(0.05,0.95,f"r={coeff_corr:.3f}",transform=ax[0].transAxes, fontsize=12)
ax[0].set_ylabel("Erreur absolue (m/s)")
ax[0].set_xlabel("Gradient de la vitesse (m/s/pixel)")
ax[0].set_title("Erreurs absolues vs gradients")
ax[0].legend()
ax[1].scatter(grad_all, err_rel_all, alpha=0.5, s=2, color='royalblue', label='Predictions')
ax[1].plot(grad_all, s * grad_all + i, color='red', linestyle='--')
ax[1].text(0.05,0.95,f"r={coeff_corr_rel:.3f}",transform=ax[1].transAxes, fontsize=12)
ax[1].set_ylabel("Erreur relative (%)")
ax[1].set_xlabel("Gradient de la vitesse (m/s/pixel)")
ax[1].set_title("Erreurs relatives vs gradients")
fig.tight_layout()
fig.suptitle("Correlation entre les erreurs et les gradients de vitesses du dataset de test")
plt.savefig(os.path.join(output_dir, "scatter_plot_mae_gradient_rel.png"), dpi=300)
plt.show()


plt.figure(figsize=(6,6))
plt.scatter(grad_all, error_all, alpha=0.5, s=2, color='royalblue', label='Predictions')
plt.ylabel("Erreur absolue (m/s)")
plt.xlabel("Gradient de la vitesse (m/s/pixel)")
plt.plot(grad_all, slope * grad_all + intercept, color='red', linestyle='--')
plt.text(0.05,0.95,f"r={coeff_corr:.3f}",transform=plt.gca().transAxes, fontsize=12)
plt.title("Erreurs absolues vs gradients sur le dataset de test")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "scatter_plot_mae_gradient.png"), dpi=300)
plt.show()


# --- Correlation linéaire pour relative errors ---
coeff_corr_rel = np.corrcoef(grad_all, err_rel_all)[0,1]
slope, intercept, r_value, p_value, std_err = stats.linregress(grad_all, err_rel_all)

plt.figure(figsize=(6,6))
plt.scatter(grad_all, err_rel_all, alpha=0.5, s=2, color='royalblue', label='Predictions')
plt.plot(grad_all, slope * grad_all + intercept, color='red', linestyle='--')
plt.text(0.05,0.95,f"r={coeff_corr_rel:.3f}",transform=plt.gca().transAxes, fontsize=12)
plt.ylabel("Erreur relative (%)")
plt.xlabel("Gradient de la vitesse (m/s/pixel)")
plt.title("Erreurs relatives vs gradients")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "scatter_plot_mae_rel_gradient.png"), dpi=300)
plt.show()








# //////////////////////////////////////
# évaluation du biais du modèle
# //////////////////////////////////////

# réalisation des prédictions sur le dataset de test

def predictions_test(dataset_test, v_min, v_max):
    pred_list, tar_list = [], []
    for i in range(dataset_test.shape[0]):
        inp = dataset_test[i:i+1]
        tar = dataset_test[i:i+1]
        pred,_ ,_ ,_ , _= prediction(inp, tar, v_min, v_max)
        pred_list.append(np.array(pred).flatten())
        tar_list.append(np.array(tar).flatten())
        pred_all = np.concatenate(pred_list)
        tar_all = np.concatenate(tar_list)
    return pred_all, tar_all 

pred_all, tar_all = predictions_test(x_test, v_min, v_max)


# tracé du scatter plot

plt.figure(figsize=(6,6))
plt.scatter(tar_all, pred_all, alpha=0.5, s=2, color='royalblue', label='Predictions')
plt.xlabel("vitesses réelles")
plt.ylabel("vitesses prédites")
# diagonale parfaite
plt.plot([np.array(tar_all).min(), np.array(tar_all).max()],
         [np.array(pred_all).min(), np.array(pred_all).max()],
         lw=2, color='red')
plt.title(f"diagramme des corrélations entre les vitesses prédites et réelles")
plt.legend()
plt.savefig(os.path.join(output_dir, "diagramme_des_correlations.png"), dpi=300)
plt.show()



















