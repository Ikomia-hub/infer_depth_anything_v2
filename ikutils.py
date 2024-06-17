import requests
import os
import torch

from infer_depth_anything_v2.DepthAnythingV2.depth_anything_v2.dpt import DepthAnythingV2

# Model configurations
model_configs = {
    'vits': {
        'encoder': 'vits',
        'features': 64,
        'out_channels': [48, 96, 192, 384],
        'model_link': 'https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth?download=true'
    },
    'vitb': {
        'encoder': 'vitb',
        'features': 128,
        'out_channels': [96, 192, 384, 768],
        'model_link': 'https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth?download=true'
    },
    'vitl': {
        'encoder': 'vitl',
        'features': 256,
        'out_channels': [256, 512, 1024, 1024],
        'model_link': 'https://huggingface.co/depth-anything/Depth-Anything-V2-/resolve/main/depth_anything_v2_vitl.pth?download=true'
    },
    'vitg': {
        'encoder': 'vitg',
        'features': 384,
        'out_channels': [1536, 1536, 1536, 1536]
    }
}

def get_model_config(name):
    """Retrieve model configuration by name."""
    return model_configs.get(name, None)

def download_model(name, models_folder):
    """Download the model if it doesn't exist locally."""
    model_config = get_model_config(name)
    if not model_config:
        print(f"No model configuration found for {name}")
        return

    URL = model_config['model_link']
    model_path = os.path.join(models_folder, f"depth_anything_v2_{name}.pth")

    if os.path.exists(model_path):
        print(f"Model for {name} already exists.")
        return

    os.makedirs(models_folder, exist_ok=True)
    print(f"Downloading model for {name} from {URL}")

    try:
        response = requests.get(URL)
        response.raise_for_status()  # Check for request errors
        with open(model_path, "wb") as f:
            f.write(response.content)
        print(f"Model for {name} downloaded successfully.")
    except requests.RequestException as e:
        print(f"Failed to download model: {e}")

def load_model(name, param, device='cpu'):
    """Load the model with specified configuration."""
    model_config = get_model_config(name)

    model_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "weights")
    model_path = os.path.join(model_folder, f'depth_anything_v2_{name}.pth')

    # Download model if it does not exist
    if not os.path.exists(model_path):
        print(f"Model file for {name} not found at {model_path}. Downloading now.")
        download_model(name, model_folder)

    model = DepthAnythingV2(
        encoder=model_config['encoder'],
        features=model_config['features'],
        out_channels=model_config['out_channels']
    )

    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.to(device).eval()

    return model
