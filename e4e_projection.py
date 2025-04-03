import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from argparse import Namespace
from e4e.models.psp import pSp

@torch.no_grad()
def projection(img, name, device='cuda'):
    """Handle both PIL and numpy array inputs"""
    # Convert numpy array to PIL if needed
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img.astype('uint8'))
    
    # Ensure we have a PIL Image
    if not isinstance(img, Image.Image):
        raise TypeError(f"Expected PIL Image or numpy array, got {type(img)}")

    model_path = 'models/e4e_ffhq_encode.pt'
    ckpt = torch.load(model_path, map_location='cpu')
    opts = ckpt['opts']
    opts['checkpoint_path'] = model_path
    opts = Namespace(**opts)
    net = pSp(opts, device).eval().to(device)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    img = transform(img).unsqueeze(0).to(device)
    images, w_plus = net(img, randomize_noise=False, return_latents=True)
    result_file = {'latent': w_plus[0]}
    torch.save(result_file, name)
    return w_plus[0]
