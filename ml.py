from pathlib import Path
import torch
from diffusers import StableDiffusionPipeline
from PIL.Image import Image
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB

### FOr a simple model with input data as a post request from the frontend###

# Loading Iris Dataset
iris = load_iris()

# Getting features and targets from the dataset
X = iris.data
Y = iris.target

# Fitting our Model on the dataset
clf = GaussianNB()
clf.fit(X,Y)

### For using model from huggingface hub ###
token_path = Path("token.txt")
token = token_path.read_text().strip()

# get your token at https://huggingface.co/settings/tokens
pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    #revision="fp16",
    #torch_dtype=torch.float16,
    use_auth_token=token,
)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
pipe.to(device)

def obtain_image(
    prompt: str,
    *,
    seed: int = 1024,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
) -> Image:
    generator = None if seed is None else torch.Generator(device).manual_seed(seed)
    print(f"Using device: {pipe.device}")
    image: Image = pipe(
        prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator,
    ).images[0]
    return image

