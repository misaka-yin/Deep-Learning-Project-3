import torch
import os, torch, torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms, datasets
import json

# -------- Preprocessing transforms --------
RAW_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])
NORM_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

# -------- Build numeric_labels (global ImageNet indices) --------
# 1) Load subdirectory names (synsets) from TestDataSet
_ds = datasets.ImageFolder("./TestDataSet", transform=RAW_TRANSFORM)
# 2) Load label list provided in labels_list.json
_entries = json.load(open("./TestDataSet/labels_list.json"))
# 3) Split each entry into (synset, index)
_pairs = [e.split(": ", 1) for e in _entries]
# 4) Map each synset to its global ImageNet index
_syn2idx = {syn: int(idx) for syn, idx in _pairs}
# 5) Build numeric label list for all 500 images
numeric_labels = [_syn2idx[s] for s in _ds.classes]


def compute_topk_accuracy(outputs: torch.Tensor,
                          targets: torch.Tensor,
                          topk=(1,5)) -> dict:
    """Return dictionary of top-k accuracy values {k: accuracy_in_%}"""
    maxk = max(topk)
    batch_size = targets.size(0)

    _, pred = outputs.topk(maxk, dim=1, largest=True, sorted=True)  # [B, maxk]
    pred = pred.t()  # [maxk, B]
    correct = pred.eq(targets.view(1, -1).expand_as(pred))

    res = {}
    for k in topk:
        # Count number of correct predictions within top-k
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res[k] = (correct_k.item() * 100.0) / batch_size
    return res

# -------- FGSM Attack --------
def fgsm_attack(model, x: torch.Tensor, y: torch.Tensor,
                epsilon: float, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """
    Perform FGSM attack. 
    x: [B, 3, 224, 224] in [0,1], y: [B] global ImageNet labels
    """
    x_adv = x.clone().detach().to(mean.device)
    x_adv.requires_grad_()
    logits = model((x_adv - mean)/std)
    loss = F.cross_entropy(logits, y)
    loss.backward()
    x_adv = torch.clamp(x_adv + epsilon * x_adv.grad.sign(), 0.0, 1.0).detach()
    return x_adv

# -------- Custom dataset for adversarial samples --------
class AdvDataset(Dataset):
    def __init__(self, image_dir, labels, transform):
        self.files = sorted(os.listdir(image_dir))
        self.dir   = image_dir
        self.labels= labels
        self.transform = transform

    def __len__(self): return len(self.labels)

    def __getitem__(self, i):
        path = os.path.join(self.dir, self.files[i])
        img  = Image.open(path).convert("RGB")
        return self.transform(img), self.labels[i]

# -------- Iterative PGD Attack --------
def pgd_attack(model: torch.nn.Module,
               x: torch.Tensor,
               y: torch.Tensor,
               epsilon: float,
               alpha: float,
               iters: int,
               mean: torch.Tensor,
               std: torch.Tensor) -> torch.Tensor:
    """
    Perform multi-step PGD attack. x ∈ [0,1], y = global ImageNet label
    """
    x_adv = x.clone().detach().to(mean.device)
    x_orig = x_adv.clone()

    for _ in range(iters):
        x_adv.requires_grad_()
        logits = model((x_adv - mean)/std)
        loss   = F.cross_entropy(logits, y)
        loss.backward()
        # Update + project to ε-ball
        x_adv = x_adv + alpha * x_adv.grad.sign()
        x_adv = torch.max(torch.min(x_adv, x_orig + epsilon), x_orig - epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()

    return x_adv

# -------- Patch-based PGD Attack --------
def pgd_patch_attack(model: torch.nn.Module,
                     x: torch.Tensor,
                     y: torch.Tensor,
                     epsilon: float,
                     alpha: float,
                     iters: int,
                     mean: torch.Tensor,
                     std: torch.Tensor,
                     patch_size: int,
                     random_patch: bool = False) -> torch.Tensor:
    """
    Perform patch-only PGD attack on x.
    If random_patch=True, choose a random 32×32 region; else center it.
    """
    x_adv = x.clone().detach().to(mean.device)
    x_orig = x_adv.clone()
    B, C, H, W = x.shape

    # Create mask to isolate the patch area
    if random_patch:
        top  = torch.randint(0, H - patch_size + 1, (1,)).item()
        left = torch.randint(0, W - patch_size + 1, (1,)).item()
    else:
        top  = (H - patch_size) // 2
        left = (W - patch_size) // 2

    mask = torch.zeros_like(x_adv)
    mask[:, :, top:top+patch_size, left:left+patch_size] = 1.0

    for _ in range(iters):
        x_adv.requires_grad_()
        logits = model((x_adv - mean)/std)
        loss   = F.cross_entropy(logits, y)
        loss.backward()
        grad = x_adv.grad.sign()
        # Update only the masked patch
        x_adv = x_adv + alpha * grad * mask
        # Project to ε-ball within the patch
        x_adv = torch.max(torch.min(x_adv, x_orig + epsilon*mask),
                          x_orig - epsilon*mask)
        x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()

    return x_adv
