import torch
import os, torch, torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms, datasets
import json

# ———— 预处理变换 ————
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

# ———— 构造 numeric_labels ————
# 1) 读取 TestDataSet 文件夹下面的子目录顺序
_ds = datasets.ImageFolder("./TestDataSet", transform=RAW_TRANSFORM)
# 2) 读取老师给的 labels_list.json
_entries = json.load(open("./TestDataSet/labels_list.json"))
# 3) 拆分成 [(synset, idx_str), ...]
_pairs = [e.split(": ", 1) for e in _entries]
# 4) synset→全局 ImageNet idx
_syn2idx = {syn: int(idx) for syn, idx in _pairs}
# 5) 对应每张样本的全局标签列表（长度500）
numeric_labels = [_syn2idx[s] for s in _ds.classes]


def compute_topk_accuracy(outputs: torch.Tensor,
                          targets: torch.Tensor,
                          topk=(1,5)) -> dict:
    """返回 {k: accuracy_in_%}"""
    maxk = max(topk)
    batch_size = targets.size(0)

    # [B, maxk]
    _, pred = outputs.topk(maxk, dim=1, largest=True, sorted=True)
    # [maxk, B]
    pred = pred.t()
    # [maxk, B]
    correct = pred.eq(targets.view(1, -1).expand_as(pred))

    res = {}
    for k in topk:
        # correct[:k] 展开成一维后统计 True 数
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res[k] = (correct_k.item() * 100.0) / batch_size
    return res

# FGSM 攻击函数
def fgsm_attack(model, x: torch.Tensor, y: torch.Tensor,
                epsilon: float, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """
    x: [B,3,224,224] ∈ [0,1], y: [B] 全局 ImageNet 标签
    """
    x_adv = x.clone().detach().to(mean.device)
    x_adv.requires_grad_()
    logits = model((x_adv - mean)/std)
    loss = F.cross_entropy(logits, y)
    loss.backward()
    x_adv = torch.clamp(x_adv + epsilon * x_adv.grad.sign(), 0.0, 1.0).detach()
    return x_adv

# 通用对抗样本 Dataset
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

def pgd_attack(model: torch.nn.Module,
               x: torch.Tensor,
               y: torch.Tensor,
               epsilon: float,
               alpha: float,
               iters: int,
               mean: torch.Tensor,
               std: torch.Tensor) -> torch.Tensor:
    """
    多步迭代 PGD 攻击。x ∈ [0,1], y ∈ 全局标签 idx.
    """
    x_adv = x.clone().detach().to(mean.device)
    x_orig = x_adv.clone()

    for _ in range(iters):
        x_adv.requires_grad_()
        logits = model((x_adv - mean)/std)
        loss   = F.cross_entropy(logits, y)
        loss.backward()
        # 投影 + 更新
        x_adv = x_adv + alpha * x_adv.grad.sign()
        x_adv = torch.max(torch.min(x_adv, x_orig + epsilon), x_orig - epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()

    return x_adv

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
    对输入 x 做 patch-only 的 PGD 攻击。
    如果 random_patch=True，会每次随机选一个 32×32 区域，否则居中。
    """
    x_adv = x.clone().detach().to(mean.device)
    x_orig = x_adv.clone()
    B, C, H, W = x.shape

    # 构造 mask
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
        # 只在 mask 区域更新
        x_adv = x_adv + alpha * grad * mask
        # 投影到 L∞ 范围内
        x_adv = torch.max(torch.min(x_adv, x_orig + epsilon*mask),
                          x_orig - epsilon*mask)
        x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()

    return x_adv
