from argparse import ArgumentParser
from tqdm import trange
from PIL import Image
import os

import torch
from torch.optim import Adam
from torch.nn.functional import mse_loss
from torchvision.transforms.functional import pil_to_tensor
from diffusers.utils import load_image

from peft import LoraConfig
from sd_utils import StableDiffusion


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def main(args):

    # Intialize diffusion model
    device = "cuda"
    model_id = "runwayml/stable-diffusion-v1-5"
    # model_id = "stabilityai/stable-diffusion-2-1"

    diffusion_model = StableDiffusion(
        device=device,
        hf_key=model_id,
        fp16=False, # TODO: use fp16
    )
    diffusion_model.get_text_embeds([args.prompt], [""])

    for p in diffusion_model.parameters():
        p.requires_grad_(False)

    # cache target embedding
    e_tgt = diffusion_model.embeddings['pos'].clone()

    diffusion_model.embeddings['pos'].requires_grad_(True)
    optimizer = Adam([diffusion_model.embeddings['pos']], lr=2e-3)

    # load image
    input_image = load_image(args.image_path).resize((512, 512))
    input_image = (pil_to_tensor(input_image).to(torch.float16) / 255.).to(device).unsqueeze(0)

    # (A) Text Embedding Optimization
    pbar = trange(args.step_a, desc='Step A', leave=True)
    for i in pbar:
        optimizer.zero_grad()
        loss = diffusion_model.train_step(
            input_image.clone().detach()
        )

        loss.backward(retain_graph=True)
        optimizer.step()

        pbar.set_description(f"Step A, loss: {loss.item():.3f}")

    # (B) Model Fine-Tuning
    # We use LoRA here for efficient fine-tuning
    unet_lora_config = LoraConfig(
        r=4,
        lora_alpha=4,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    diffusion_model.unet.add_adapter(unet_lora_config)
    lora_layers = filter(lambda p: p.requires_grad, diffusion_model.unet.parameters())
    optimizer = Adam(lora_layers, lr=1e-4)

    pbar = trange(args.step_b, desc='Step B', leave=True)
    for i in pbar:
        optimizer.zero_grad()
        loss = diffusion_model.train_step(
            input_image.clone().detach()
        )
        loss.backward(retain_graph=True)
        optimizer.step()

        pbar.set_description(f"Step B, loss: {loss.item():.5f}")

    # (C) Interpolation & Generation
    NUM_SAMPLES = 20
    with torch.no_grad():
        for i in range(NUM_SAMPLES):
            interp_strength = i / (NUM_SAMPLES - 1)
            e_interp = e_tgt * interp_strength + diffusion_model.embeddings['pos'] * (1 - interp_strength)
            output = diffusion_model.prompt_to_img(
                e_interp
            )
            output_image = Image.fromarray(output[0])
            output_image.save(f'{args.savedir}/C_{interp_strength}.png')

    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True,
                        help="target image path")
    parser.add_argument("--prompt", type=str, required=True,
                        help="target prompt")
    parser.add_argument("--savedir", type=str, required=True,
                        help="save directory for output images")
    parser.add_argument("--step_a", type=int, default=1000,
                        help="save directory for output images")
    parser.add_argument("--step_b", type=int, default=1500,
                        help="save directory for output images")
    args = parser.parse_args()

    os.makedirs(args.savedir, exist_ok=True)

    seed_everything(42)
    main(args)