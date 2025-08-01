import os
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import torch 
import os
import torch.nn.functional as F
import numpy as np
import segmentation_models_pytorch as smp
import torch.nn as nn
import torch.optim as optim
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as T
import pickle
from torchvision import transforms
import matplotlib.pyplot as plt

# ------------------------------ #
# 1. Load and Align Softmax Maps #
# ------------------------------ #

def load_and_align_avg_probs(prob_dirs_256, prob_dirs_512, sample_index, target_size=(512, 512)):
    probs_256 = []
    for d in prob_dirs_256:
        path = os.path.join(d, f"prob_{sample_index:04d}.npy")
        prob = np.load(path)
        tensor = torch.from_numpy(prob).unsqueeze(0)  # [1, C, H, W]
        resized = F.interpolate(tensor, size=target_size, mode='bilinear', align_corners=False)
        probs_256.append(resized.squeeze(0))  # [C, 512, 512]
    avg_prob_256 = torch.stack(probs_256).mean(dim=0)

    probs_512 = []
    for d in prob_dirs_512:
        path = os.path.join(d, f"prob_{sample_index:04d}.npy")
        prob = np.load(path)
        tensor = torch.from_numpy(prob)
        probs_512.append(tensor)  # [C, 512, 512]
    avg_prob_512 = torch.stack(probs_512).mean(dim=0)

    return avg_prob_256, avg_prob_512

# --------------------------------------- #
# 2. Entropy-Based Explainability Methods #
# --------------------------------------- #

def compute_entropy_map(prob_fused):
    eps = 1e-8
    entropy = -torch.sum(prob_fused * torch.log(prob_fused + eps), dim=0)  # [H, W]
    return entropy

def get_high_entropy_mask(entropy_map, top_percent=20):
    threshold = torch.quantile(entropy_map, 1 - top_percent / 100.0)
    return (entropy_map >= threshold).float()


def visualize_and_save_entropy_on_image(image_tensor, entropy_map, label_tensor, save_path, alpha=0.5):
    image_np = image_tensor.squeeze().cpu().numpy()
    if image_np.ndim == 2:
        image_np = np.stack([image_np] * 3, axis=-1)  # [H, W, 3]
    else:
        image_np = np.transpose(image_np, (1, 2, 0))  # [H, W, 3]

    entropy_np = entropy_map.cpu().numpy()
    label_np = label_tensor.squeeze().cpu().numpy()

    # Mask background (assume class 0 = background)
    foreground_mask = label_np != 0
    entropy_masked = np.where(foreground_mask, entropy_np, np.nan)  # use np.nan to ignore background in colorbar scale

    entropy_vis = (entropy_masked - np.nanmin(entropy_masked)) / (np.nanmax(entropy_masked) - np.nanmin(entropy_masked) + 1e-8)

    plt.figure(figsize=(6, 6))
    plt.imshow(image_np, cmap='gray')
    plt.imshow(entropy_vis, cmap='jet', alpha=alpha)
    plt.colorbar(label='Entropy (masked)')
    plt.title("Entropy Map Overlay (Foreground Only)")
    plt.axis('off')
    plt.tight_layout()

    overlay_path = save_path + "_entropy_overlay_masked.png"
    plt.savefig(overlay_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"Saved overlay (masked) to: {overlay_path}")

    entropy_img = np.nan_to_num(entropy_vis, nan=0.0)
    entropy_img = (entropy_img * 255).astype(np.uint8)
    Image.fromarray(entropy_img).save(save_path + "_entropy_gray_masked.png")
    np.save(save_path + "_entropy_masked.npy", entropy_masked)


# ------------------------ #
# 3. Visualization Pipeline #
# ------------------------ #

def fuse_probs_and_visualize_entropy(test_loader_512, prob_dirs_256, prob_dirs_512, output_dir="/home/feg48/2.5D_seg/entropy_vis_outputs"):
    os.makedirs(output_dir, exist_ok=True)

    # for idx, (x, _) in enumerate(test_loader_512):  # x: [B, C, H, W]
    for idx, (x, label) in enumerate(test_loader_512):  # include labels
        batch_size = x.size(0)

        for i in range(batch_size):
            sample_index = idx * batch_size + i
            print(f"Processing sample {sample_index}")

            # Load and fuse softmax
            prob_256, prob_512 = load_and_align_avg_probs(prob_dirs_256, prob_dirs_512, sample_index)
            fused_prob = (prob_256 + prob_512) / 2  # [C, 512, 512]

            # Compute entropy and save visualization
            entropy_map = compute_entropy_map(fused_prob)
            save_path = os.path.join(output_dir, f"sample_{sample_index:04d}")
            # visualize_and_save_entropy_on_image(x[i], entropy_map, save_path)
            visualize_and_save_entropy_on_image(x[i], entropy_map, label[i], save_path)


# DataLoader
train_val_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

test_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

label_transform = transforms.Compose([
        transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.PILToTensor(),
    ])

class KneeSegmentation25D(Dataset):
    def __init__(self, image_dir, mask_dir, filenames, img_transforms, label_transforms):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.filenames = sorted(filenames)
        self.img_transforms = img_transforms
        self.label_transforms = label_transforms


    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        study_id, slice_tag = filename.replace(".jpg", "").split("_slice_")
        slice_num = int(slice_tag)

        stack = []
        for offset in [-1, 0, 1]:
            n = slice_num + offset
            neighbor_file = f"{study_id}_slice_{n:03d}.jpg"
            neighbor_path = os.path.join(self.image_dir, neighbor_file)
            if os.path.exists(neighbor_path):
                img = Image.open(neighbor_path).convert("L")
            else:
                img = Image.open(os.path.join(self.image_dir, filename)).convert("L")

            img = self.img_transforms(img)
            img = img.squeeze(0) 
            stack.append(img)
            
        image = np.stack(stack, axis=0)  # shape: (3, H, W)


        mask_path = os.path.join(self.mask_dir, study_id, filename.replace(".jpg", ".npy"))
        mask = np.load(mask_path).astype(np.int64)
        mask = Image.fromarray(mask.astype(np.uint8))
        mask_resized = self.label_transforms(mask)
        
        return image, mask_resized.clone().long()

#Load saved split and recreate test_loader_512
split_save_path_512 = "/data_vault/hexai/OAICartilage/knee_split_512_noise_notebook.pkl"

with open(split_save_path_512, "rb") as f:
    split_info = pickle.load(f)

test_f = split_info["test"]

image_dir_512 = "/data_vault/hexai/OAICartilage/image_manual_crops"   
mask_dir_512 = "/data_vault/hexai/OAICartilage/cropped_annotations_numpy"  
batch_size = 8  

test_ds_512 = KneeSegmentation25D(image_dir_512, mask_dir_512, test_f, test_transform, label_transform)
test_loader_512 = DataLoader(test_ds_512, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True)

print(f"Loaded test_loader_512 with {len(test_ds_512)} samples.")

# ----------------------- #
# 4. Setup Directories #
# ----------------------- #

save_dirs_256 = [
    "/home/feg48/2.5D_seg/256_probs_5_0.15_17_good",
    "/home/feg48/2.5D_seg/256_probs_7_0.1_17_good"
]

save_dirs_512 = [
    "/home/feg48/2.5D_seg/512_probs_7_0.1_1337_good",
    "/home/feg48/2.5D_seg/512_probs_5_0.15_17_good"
]

#Call function
fuse_probs_and_visualize_entropy(test_loader_512, save_dirs_256, save_dirs_512)
