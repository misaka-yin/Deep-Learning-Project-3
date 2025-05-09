import torch

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
