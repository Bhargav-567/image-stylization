import os
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import face_recognition
from PIL import Image
import torchvision.transforms as transforms
from matplotlib import pyplot as plt

# Google Drive paths for model downloads
google_drive_paths = {
    "models/stylegan2-ffhq-config-f.pt": "https://drive.google.com/uc?id=1Yr7KuD959btpmcKGAUsbAk5rPjX2MytK",
    "models/dlibshape_predictor_68_face_landmarks.dat": "https://drive.google.com/uc?id=11BDmNKS1zxSZxkgsEvQoKgFd8J264jKp",
    "models/e4e_ffhq_encode.pt": "https://drive.google.com/uc?id=1o6ijA3PkcewZvwJJ73dJ0fxhndn0nnh7",
    "models/restyle_psp_ffhq_encode.pt": "https://drive.google.com/uc?id=1nbxCIVw9H3YnQsoIPykNEFwWJnHVHlVd",
    "models/arcane_caitlyn.pt": "https://drive.google.com/uc?id=1gOsDTiTPcENiFOrhmkkxJcTURykW1dRc",
    "models/arcane_caitlyn_preserve_color.pt": "https://drive.google.com/uc?id=1cUTyjU-q98P75a8THCaO545RTwpVV-aH",
    "models/arcane_jinx_preserve_color.pt": "https://drive.google.com/uc?id=1jElwHxaYPod5Itdy18izJk49K1nl4ney",
    "models/arcane_jinx.pt": "https://drive.google.com/uc?id=1quQ8vPjYpUiXM4k1_KIwP4EccOefPpG_",
    "models/disney.pt": "https://drive.google.com/uc?id=1zbE2upakFUAx8ximYnLofFwfT8MilqJA",
    "models/disney_preserve_color.pt": "https://drive.google.com/uc?id=1Bnh02DjfvN_Wm8c4JdOiNV4q9J7Z_tsi",
    "models/jojo.pt": "https://drive.google.com/uc?id=13cR2xjIBj8Ga5jMO7gtxzIJj2PDsBYK4",
    "models/jojo_preserve_color.pt": "https://drive.google.com/uc?id=1ZRwYLRytCEKi__eT2Zxv1IlV6BGVQ_K2",
    "models/jojo_yasuho.pt": "https://drive.google.com/uc?id=1grZT3Gz1DLzFoJchAmoj3LoM9ew9ROX_",
    "models/jojo_yasuho_preserve_color.pt": "https://drive.google.com/uc?id=1SKBu1h0iRNyeKBnya_3BBmLr4pkPeg_L",
    "models/supergirl.pt": "https://drive.google.com/uc?id=1L0y9IYgzLNzB-33xTpXpecsKU-t9DpVC",
    "models/supergirl_preserve_color.pt": "https://drive.google.com/uc?id=1VmKGuvThWHym7YuayXxjv0fSn32lfDpE",
}

@torch.no_grad()
def load_model(generator, model_file_path):
    ensure_checkpoint_exists(model_file_path)
    ckpt = torch.load(model_file_path, map_location=lambda storage, loc: storage)
    generator.load_state_dict(ckpt["g_ema"], strict=False)
    return generator.mean_latent(50000)

def ensure_checkpoint_exists(model_weights_filename):
    if not os.path.isfile(model_weights_filename) and model_weights_filename in google_drive_paths:
        gdrive_url = google_drive_paths[model_weights_filename]
        try:
            from gdown import download as drive_download
            drive_download(gdrive_url, model_weights_filename, quiet=False)
        except ModuleNotFoundError:
            print("gdown module not found. Install it using: pip install gdown")
    
    if not os.path.isfile(model_weights_filename):
        print(f"{model_weights_filename} not found. Please download it manually.")

@torch.no_grad()
def load_source(files, generator, device='cuda'):
    sources = []
    for file in files:
        source = torch.load(f'./inversion_codes/{file}.pt')['latent'].to(device)
        if source.ndim == 3:
            source = generator.get_latent(source, truncation=1, is_latent=True)
            source = list2style(source)
        sources.append(source)
    return style2list(torch.cat(sources, 0)) if not isinstance(sources, list) else sources

def display_image(image, size=None, mode='nearest', title=''):
    if not isinstance(image, torch.Tensor):
        image = transforms.ToTensor()(image).unsqueeze(0)
    if image.is_cuda:
        image = image.cpu()
    if size is not None and image.size(-1) != size:
        image = F.interpolate(image, size=(size, size), mode=mode)
    if image.dim() == 4:
        image = image[0]
    image = image.permute(1, 2, 0).detach().numpy()
    plt.figure()
    plt.title(title)
    plt.axis('off')
    plt.imshow(image)

def get_landmark(filepath):
    img = face_recognition.load_image_file(filepath)
    face_landmarks_list = face_recognition.face_landmarks(img)
    if not face_landmarks_list:
        raise ValueError("Face not detected, try another image")
    lm = [point for key in face_landmarks_list[0] for point in face_landmarks_list[0][key]]
    return np.array(lm)

def align_face(filepath, output_size=1024):
    img = cv2.imread(filepath)
    if img is None:
        raise ValueError("Failed to load image")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_landmarks_list = face_recognition.face_landmarks(img_rgb)
    if not face_landmarks_list:
        raise ValueError("Face not detected, try another image")
    left_eye = np.mean(face_landmarks_list[0]["left_eye"], axis=0)
    right_eye = np.mean(face_landmarks_list[0]["right_eye"], axis=0)
    dy, dx = right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))
    eyes_center = (int((left_eye[0] + right_eye[0]) // 2), int((left_eye[1] + right_eye[1]) // 2))
    M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
    aligned_face = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_CUBIC)
    return cv2.resize(aligned_face, (output_size, output_size))

def strip_path_extension(path):
    return os.path.splitext(path)[0]
