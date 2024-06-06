import torch

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def to_device(obj, device=DEVICE):
    if isinstance(obj, dict):
        return {k: v.to(device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [x.to(device) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(x.to(device) for x in obj)
    else:
        return obj.to(device)

