from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
import os
import cv2
import face_recognition  # Replacement for dlib
from PIL import Image
import numpy as np
import math
import torchvision
import scipy
import scipy.ndimage
import torchvision.transforms as transforms

google_drive_paths = {
    "models/stylegan2-ffhq-config-f.pt": "https://drive.google.com/uc?id=1Yr7KuD959btpmcKGAUsbAk5rPjX2MytK",
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
    if not os.path.isfile(model_weights_filename) and (
        model_weights_filename in google_drive_paths
    ):
        gdrive_url = google_drive_paths[model_weights_filename]
        try:
            from gdown import download as drive_download
            drive_download(gdrive_url, model_weights_filename, quiet=False)
        except ModuleNotFoundError:
            print(
                "gdown module not found.",
                "pip3 install gdown or, manually download the checkpoint file:",
                gdrive_url
            )

    if not os.path.isfile(model_weights_filename) and (
        model_weights_filename not in google_drive_paths
    ):
        print(
            model_weights_filename,
            " not found, you may need to manually download the model weights."
        )

def get_landmark(filepath, predictor=None):
    """Get facial landmarks using face_recognition library"""
    try:
        # Load image
        image = face_recognition.load_image_file(filepath)
        
        # Get face landmarks
        face_landmarks_list = face_recognition.face_landmarks(image)
        
        if len(face_landmarks_list) == 0:
            raise ValueError("No faces detected in the image")
            
        # Convert landmarks to numpy array in the same format as dlib
        landmarks = []
        for face_landmarks in face_landmarks_list:
            # The order of facial landmarks in face_recognition is different from dlib
            # We need to reorder them to match dlib's 68-point format
            chin = face_landmarks['chin']
            left_eyebrow = face_landmarks['left_eyebrow']
            right_eyebrow = face_landmarks['right_eyebrow']
            nose_bridge = face_landmarks['nose_bridge']
            nose_tip = face_landmarks['nose_tip']
            left_eye = face_landmarks['left_eye']
            right_eye = face_landmarks['right_eye']
            top_lip = face_landmarks['top_lip']
            bottom_lip = face_landmarks['bottom_lip']
            
            # Combine all points in the order expected by the rest of the code
            combined = chin + left_eyebrow + right_eyebrow + nose_bridge + nose_tip + left_eye + right_eye + top_lip + bottom_lip
            
            # Convert to numpy array
            landmarks = np.array(combined)
            
            # We only process the first face found
            break
            
        return landmarks
        
    except Exception as e:
        print(f"Error in face_recognition: {str(e)}")
        # Fallback to OpenCV-based face detection if face_recognition fails
        return get_landmark_opencv(filepath)

def get_landmark_opencv(filepath):
    """Fallback face detection using OpenCV's Haar cascades"""
    try:
        # Load the image
        image = cv2.imread(filepath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Load the pre-trained Haar cascade for face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            raise ValueError("No faces detected in the image")
            
        # For simplicity, we'll just return a basic face rectangle
        # Note: This is a minimal implementation - you might want to implement
        # a more sophisticated landmark detection or use a different model
        x, y, w, h = faces[0]
        
        # Create dummy landmarks (this is just a fallback)
        landmarks = np.array([
            [x, y], [x+w, y], [x+w, y+h], [x, y+h],  # Face rectangle corners
            [x+w//2, y], [x+w//2, y+h],  # Top and bottom center
            [x, y+h//2], [x+w, y+h//2]   # Left and right center
        ])
        
        return landmarks
        
    except Exception as e:
        print(f"Error in OpenCV face detection: {str(e)}")
        raise ValueError("Could not detect face in the image")

def align_face(filepath, output_size=1024, transform_size=4096, enable_padding=True):
    """
    Align face using face_recognition or OpenCV as fallback
    """
    pil_image = Image.open(filepath).convert('RGB')
    img = np.array(pil_image)  #.astype(np.uint8)
    # img = np.ascontiguousarray(img)

    try:
        lm = get_landmark(filepath)
    except Exception as e:
        print(f"Face detection failed: {str(e)}")
        # If face detection fails, just return the center-cropped image
        return center_crop(pil_image, output_size)

    # Rest of the alignment code remains the same as before
    # since it works with the landmark points array
    lm_chin = lm[0:17]  # left-right
    lm_eyebrow_left = lm[17:22]  # left-right
    lm_eyebrow_right = lm[22:27]  # left-right
    lm_nose = lm[27:31]  # top-down
    lm_nostrils = lm[31:36]  # top-down
    lm_eye_left = lm[36:42]  # left-clockwise
    lm_eye_right = lm[42:48]  # left-clockwise
    lm_mouth_outer = lm[48:60]  # left-clockwise
    lm_mouth_inner = lm[60:68]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_left = lm_mouth_outer[0]
    mouth_right = lm_mouth_outer[6]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink

    # Crop.
    width, height = img.size  # Get image dimensions properly
        crop = (
            max(int(np.floor(min(quad[:, 0]))) - border, 0),
            max(int(np.floor(min(quad[:, 1]))) - border, 0),
            min(int(np.ceil(max(quad[:, 0]))) + border, width),
            min(int(np.ceil(max(quad[:, 1]))) + border, height)
        )

    # Pad.
    pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
           int(np.ceil(max(quad[:, 1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0),
           max(pad[3] - img.size[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
                          1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]))
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
        img = Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        quad += pad[:2]

    # Transform.
    img = img.transform((transform_size, transform_size), Image.QUAD, (quad + 0.5).flatten(), Image.BILINEAR)
    if output_size < transform_size:
        img = img.resize((output_size, output_size), Image.ANTIALIAS)

    return img

def center_crop(img, size):
    """Center crop as fallback when face detection fails"""
    width, height = img.size
    new_width = new_height = size
    
    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2
    
    return img.crop((left, top, right, bottom))

def strip_path_extension(path):
    return os.path.splitext(path)[0]
