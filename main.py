import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple, Optional, Union

# Add taming-transformers to path
current_dir = os.path.dirname(os.path.abspath(__file__))
taming_path = os.path.join(current_dir, 'taming-transformers')
sys.path.append(taming_path)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch import optim
from torchvision.transforms import InterpolationMode

import numpy as np
from PIL import Image, ImageFile
from omegaconf import OmegaConf
from tqdm import tqdm

from CLIP import clip
import kornia.augmentation as K
from taming.models import cond_transformer, vqgan

ImageFile.LOAD_TRUNCATED_IMAGES = True

class ReplaceGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_forward, x_backward):
        ctx.shape = x_backward.shape
        return x_forward

    @staticmethod
    def backward(ctx, grad_in):
        return None, grad_in.reshape(ctx.shape)

replace_grad = ReplaceGrad.apply

class ClampWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, min, max):
        ctx.min = min
        ctx.max = max
        ctx.save_for_backward(input)
        return input.clamp(min, max)

    @staticmethod
    def backward(ctx, grad_in):
        input, = ctx.saved_tensors
        return grad_in * (grad_in * (input - input.clamp(ctx.min, ctx.max)) >= 0), None, None

clamp_with_grad = ClampWithGrad.apply

class MakeCutouts(nn.Module):
    def __init__(self, cut_size: int, cutn: int, cut_pow: float = 1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow
        self.augs = nn.Sequential(
            K.RandomHorizontalFlip(p=0.5),
            K.ColorJitter(hue=0.01, saturation=0.01, p=0.7)
        )
        self.noise_fac = 0.1

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        
        for _ in range(self.cutn):
            size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutouts.append(F.interpolate(cutout, (self.cut_size, self.cut_size), 
                                      mode='bilinear', align_corners=True))
        
        batch = self.augs(torch.cat(cutouts, dim=0))
        if self.noise_fac:
            noise = torch.randn_like(batch) * self.noise_fac
            batch = batch + noise
        return batch

class Prompt(nn.Module):
    def __init__(self, embed: torch.Tensor, weight: float = 1., stop: float = float('-inf')):
        super().__init__()
        self.register_buffer('embed', embed)
        self.register_buffer('weight', torch.as_tensor(weight))
        self.register_buffer('stop', torch.as_tensor(stop))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Process in smaller batches
        batch_size = 8
        input_batches = input.split(batch_size)
        results = []
        
        for batch in input_batches:
            input_normed = F.normalize(batch.unsqueeze(1), dim=2)
            embed_normed = F.normalize(self.embed.unsqueeze(0), dim=2)
            dists = input_normed.sub(embed_normed).norm(dim=2).div(2).arcsin().pow(2).mul(2)
            dists = dists * self.weight.sign()
            results.append(dists)
        
        dists = torch.cat(results, dim=0)
        return self.weight.abs() * replace_grad(dists, torch.maximum(dists, self.stop)).mean()

class VQGAN_CLIP_Generator:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        print("Loading CLIP model...")
        self.perceptor = clip.load(args.clip_model, jit=False)[0].eval().requires_grad_(False).to(self.device)
        
        print("Loading VQGAN model...")
        self.model = self.load_vqgan_model(args.vqgan_config, args.vqgan_checkpoint).to(self.device)
        
        self.cut_size = self.perceptor.visual.input_resolution
        self.make_cutouts = MakeCutouts(self.cut_size, args.cutn, cut_pow=args.cut_pow)
        self.normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                           std=[0.26862954, 0.26130258, 0.27577711])

    def load_vqgan_model(self, config_path: str, checkpoint_path: str) -> nn.Module:
        from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
        import torch.serialization
        torch.serialization.add_safe_globals([ModelCheckpoint])

        config = OmegaConf.load(config_path)
        if config.model.target == 'taming.models.vqgan.VQModel':
            model = vqgan.VQModel(**config.model.params)
            model.eval().requires_grad_(False)
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            sd = checkpoint["state_dict"]
            model.load_state_dict(sd, strict=False)
        elif config.model.target == 'taming.models.cond_transformer.Net2NetTransformer':
            parent_model = cond_transformer.Net2NetTransformer(**config.model.params)
            parent_model.eval().requires_grad_(False)
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            sd = checkpoint["state_dict"]
            parent_model.load_state_dict(sd, strict=False)
            model = parent_model.first_stage_model
        else:
            raise ValueError(f'Unknown model type: {config.model.target}')
        
        return model

    def vector_quantize(self, x):
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()  # [B, C, H, W] -> [B, H, W, C]
        
        codebook = self.model.quantize.embedding.weight
        d = x.pow(2).sum(dim=-1, keepdim=True) + \
            codebook.pow(2).sum(dim=1) - 2 * torch.einsum('bhwc,nc->bhwn', x, codebook)
        indices = d.argmin(-1)
        x_q = F.one_hot(indices, codebook.shape[0]).to(x.dtype) @ codebook
        
        # Return to original shape
        x_q = x_q.permute(0, 3, 1, 2).contiguous()  # [B, H, W, C] -> [B, C, H, W]
        return replace_grad(x_q, x.permute(0, 3, 1, 2).contiguous())

    def generate(self, text_prompts: List[str], image_prompts: List[str], 
                output_dir: str = "outputs", num_iterations: int = 500):
        print("Initializing generation...")
        os.makedirs(output_dir, exist_ok=True)
        
        print("Setting up model dimensions...")
        f = 2**(self.model.decoder.num_resolutions - 1)
        toksX, toksY = self.args.size[0] // f, self.args.size[1] // f
        sideX, sideY = toksX * f, toksY * f
        
        print(f"Image size: {sideX}x{sideY}")
        print(f"Token size: {toksX}x{toksY}")

        print("Initializing latent space...")
        if self.args.init_image:
            init = Image.open(self.args.init_image).convert('RGB')
            init = init.resize((sideX, sideY), Image.LANCZOS)
            init = TF.to_tensor(init).to(self.device).unsqueeze(0) * 2 - 1
            z, *_ = self.model.encode(init)
        else:
            e_dim = self.model.quantize.e_dim
            n_toks = self.model.quantize.n_e
            z = torch.randn(1, e_dim, toksY, toksX, device=self.device)
        
        z = z.detach().clone()
        z.requires_grad_(True)
        
        opt = optim.Adam([z], lr=self.args.step_size)
        
        print("Setting up prompts...")
        pMs = self.setup_prompts(text_prompts, image_prompts)
        
        print("Starting generation loop...")
        start_time = time.time()
        try:
            with tqdm(range(num_iterations)) as pbar:
                for i in pbar:
                    opt.zero_grad(set_to_none=True)
                    
                    # Decode latent
                    z_q = self.vector_quantize(z)
                    out = clamp_with_grad(self.model.decode(z_q).add(1).div(2), 0, 1)
                    
                    # Get CLIP embeddings
                    image_embeds = self.perceptor.encode_image(
                        self.normalize(self.make_cutouts(out))).float()
                    
                    # Calculate losses
                    losses = []
                    for prompt in pMs:
                        losses.append(prompt(image_embeds))
                    
                    loss = sum(losses)
                    loss.backward()
                    opt.step()
                    
                    if i % self.args.save_freq == 0:
                        img = TF.to_pil_image(out[0].cpu())
                        img.save(os.path.join(output_dir, f'progress_{i:04d}.png'))
                    
                    if i % 10 == 0:
                        elapsed = time.time() - start_time
                        iter_per_sec = (i + 1) / elapsed if i > 0 else 0
                        pbar.set_description(f'Loss: {loss.item():.4f}, {iter_per_sec:.2f}it/s')
                    
        except KeyboardInterrupt:
            print("\nInterrupted by user. Saving final result...")
            img = TF.to_pil_image(out[0].cpu())
            img.save(os.path.join(output_dir, 'interrupted.png'))

    def parse_prompt(self, prompt: str) -> Tuple[str, float, float]:
        vals = prompt.rsplit(':', 2)
        vals = vals + ['', '1', '-inf'][len(vals):]
        return vals[0], float(vals[1]), float(vals[2])

    def setup_prompts(self, text_prompts: List[str], image_prompts: List[str]) -> List[Prompt]:
        pMs = []
        print("Processing text prompts:", text_prompts)
        for prompt in text_prompts:
            txt, weight, stop = self.parse_prompt(prompt)
            embed = self.perceptor.encode_text(clip.tokenize(txt).to(self.device)).float()
            pMs.append(Prompt(embed, weight, stop).to(self.device))

        if image_prompts:
            print("Processing image prompts:", image_prompts)
            for prompt in image_prompts:
                path, weight, stop = self.parse_prompt(prompt)
                img = Image.open(path).convert('RGB')
                img = TF.resize(img, [self.args.size[0], self.args.size[1]], 
                            interpolation=InterpolationMode.LANCZOS)
                batch = self.make_cutouts(TF.to_tensor(img).unsqueeze(0).to(self.device))
                embed = self.perceptor.encode_image(self.normalize(batch)).float()
                pMs.append(Prompt(embed, weight, stop).to(self.device))

        return pMs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompts', type=str, nargs='+', default=['a masterpiece'],
                      help='Text prompts to generate from')
    parser.add_argument('--image_prompts', type=str, nargs='+', default=[],
                      help='Image prompts to generate from')
    parser.add_argument('--size', type=int, nargs=2, default=[512, 512],
                      help='Output image size (width, height)')
    parser.add_argument('--init_image', type=str,
                      help='Path to initial image (optional)')
    parser.add_argument('--output_dir', type=str, default='outputs',
                      help='Directory to save outputs')
    parser.add_argument('--num_iterations', type=int, default=500,
                      help='Number of iterations')
    parser.add_argument('--save_freq', type=int, default=1,
                      help='Save frequency')
    parser.add_argument('--seed', type=int,
                      help='Random seed')
    
    # Model parameters
    parser.add_argument('--vqgan_config', type=str, default='vqgan_imagenet_f16_16384.yaml')
    parser.add_argument('--vqgan_checkpoint', type=str, default='vqgan_imagenet_f16_16384.ckpt')
    parser.add_argument('--clip_model', type=str, default='ViT-B/32')
    parser.add_argument('--step_size', type=float, default=0.1)
    parser.add_argument('--cutn', type=int, default=32)
    parser.add_argument('--cut_pow', type=float, default=1.)

    args = parser.parse_args()
    
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    generator = VQGAN_CLIP_Generator(args)
    generator.generate(args.prompts, args.image_prompts, 
                      args.output_dir, args.num_iterations)

if __name__ == "__main__":
    main()
