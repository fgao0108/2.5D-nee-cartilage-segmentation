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

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

# Dice Score Function
def dice_score(pred, target, num_classes=5):
    dices = []
    for cls in range(1, num_classes):  # skip background
        pred_cls = (pred == cls).float()
        target_cls = (target == cls).float()
        intersection = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum()
        dice = (2 * intersection + 1e-6) / (union + 1e-6)
        dices.append(dice.item())
    return np.mean(dices)

# IoU Score Function
def iou_score(pred, target, num_classes=5):
    ious = []
    for cls in range(1, num_classes):  # skip background
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        intersection = (pred_cls & target_cls).sum().item()
        union = (pred_cls | target_cls).sum().item()
        ious.append(intersection / union if union != 0 else 1.0)
    return np.mean(ious)


# Load and align average of multiple prob files for one sample
def load_and_align_avg_probs(prob_dirs_256, prob_dirs_512, sample_index, target_size=(512, 512)):
    # Load and resize 256x256 probs
    probs_256 = []
    for d in prob_dirs_256:
        path = os.path.join(d, f"prob_{sample_index:04d}.npy")
        prob = np.load(path)  # [C, 256, 256]
        tensor = torch.from_numpy(prob).unsqueeze(0)  # [1, C, H, W]
        resized = F.interpolate(tensor, size=target_size, mode='bilinear', align_corners=False)
        probs_256.append(resized.squeeze(0))
    avg_prob_256 = torch.stack(probs_256).mean(dim=0)  # [C, 512, 512]

    # Load 512x512 probs
    probs_512 = []
    for d in prob_dirs_512:
        path = os.path.join(d, f"prob_{sample_index:04d}.npy")
        prob = np.load(path)  # [C, 512, 512]
        tensor = torch.from_numpy(prob)
        probs_512.append(tensor)
    avg_prob_512 = torch.stack(probs_512).mean(dim=0)  # [C, 512, 512]

    return avg_prob_256, avg_prob_512

# Evaluation loop with fusion
def fuse_probs_and_evaluate(test_loader, prob_dirs_256, prob_dirs_512, num_classes=5):
    ious = []
    dices = []

    for idx, (_, y) in enumerate(test_loader):
        batch_size = y.size(0)

        for i in range(batch_size):
            sample_index = idx * batch_size + i
            
            print("5")
            prob_256_resized, prob_512 = load_and_align_avg_probs(prob_dirs_256, prob_dirs_512, sample_index)
            fused_prob = (prob_256_resized + prob_512) / 2  # [C, 512, 512]
            pred = torch.argmax(fused_prob, dim=0)  # [512, 512]
            target = y[i].cpu()
            print("6")
            ious.append(iou_score(pred, target, num_classes))
            dices.append(dice_score(pred, target, num_classes))

    mean_iou = np.mean(ious)
    mean_dice = np.mean(dices)

    print(f"Fused Test IoU:  {mean_iou:.4f}")
    print(f"Fused Test Dice: {mean_dice:.4f}")

    return mean_iou, mean_dice

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

# Manually specify these based on your dataset organization and batch size
image_dir_512 = "/data_vault/hexai/OAICartilage/image_manual_crops"   
mask_dir_512 = "/data_vault/hexai/OAICartilage/cropped_annotations_numpy"  
batch_size = 8  

test_ds_512 = KneeSegmentation25D(image_dir_512, mask_dir_512, test_f, test_transform, label_transform)
test_loader_512 = DataLoader(test_ds_512, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True)

print(f"Loaded test_loader_512 with {len(test_ds_512)} samples.")



# Then call fusion and evaluation function
save_dirs_256 = ["/home/feg48/2.5D_seg/256_probs_5_0.15_17_good", "/home/feg48/2.5D_seg/256_probs_7_0.1_17_good"]
save_dirs_512 = ["/home/feg48/2.5D_seg/512_probs_7_0.1_1337_good", "/home/feg48/2.5D_seg/512_probs_5_0.15_17_good"]


fuse_probs_and_evaluate(test_loader_512, save_dirs_256, save_dirs_512)

