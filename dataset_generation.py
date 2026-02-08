from omegaconf import OmegaConf
import torch
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision import transforms
import os
import cv2
from tqdm import tqdm
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config, default
from ldm.modules.embedding_manager import EmbeddingManager
import random
import glob
import insightface
from sklearn import preprocessing
from datetime import datetime
from people import people_attributes
from torchvision.utils import save_image

from face_data.names3 import people_attributes


# Util Functions
def load_model_from_config(config, ckpt, device="cpu", verbose=False):
    """Loads a model from config and a ckpt
    if config is a path will use omegaconf to load
    """
    if isinstance(config, (str, Path)):
        config = OmegaConf.load(config)

    pl_sd = torch.load(ckpt, map_location="cpu")
    # global_step = pl_sd["global_step"]
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    model.cond_stage_model.device = device
    return model

def load_img(path, target_size=512):
    """Load an image, resize and output -1..1"""
    image = Image.open(path).convert("RGB")


    tform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
        ])
    image = tform(image)
    return 2.*image - 1.


def get_models(config_path, ckpt_path, devices):
    model = load_model_from_config(config_path, ckpt_path, devices[0])
    sampler = DDIMSampler(model)
    return model, sampler

@torch.no_grad()
def sample_model(model, sampler, c, h, w, ddim_steps, scale, ddim_eta, start_code=None, n_samples=1, t_start=None,log_every_t=None,till_T=None,verbose=True):
    """Sample the model"""
    uc = None
    negative = '(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), cropped, out of frame, low quality, ugly, duplicate, morbid, mutilated, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions.'

    if scale != 1.0:
        uc = model.get_learned_conditioning(n_samples * [negative])
    log_t = 100
    if log_every_t is not None:
        log_t = log_every_t
    shape = [4, h // 8, w // 8]
    samples_ddim, inters = sampler.sample(S=ddim_steps,
                                     conditioning=c,
                                     batch_size=n_samples,
                                     shape=shape,
                                     verbose=False,
                                     x_T=start_code,
                                     unconditional_guidance_scale=scale,
                                     unconditional_conditioning=uc,
                                     eta=ddim_eta,
                                     verbose_iter = verbose,
                                     log_every_t = log_t,
                                     t_start=t_start,
                                     till_T=till_T
                                    )
    return samples_ddim

def set_embedding_manager(model, placeholder_strings):
    embedder = model.cond_stage_model
    num_vectors_per_token = 10
    embedding_manager = EmbeddingManager(embedder=embedder, placeholder_strings=placeholder_strings, num_vectors_per_token=num_vectors_per_token)
    for param in embedding_manager.embedding_parameters():
        param.requires_grad = True
    parameters = list(embedding_manager.embedding_parameters())
    return embedding_manager, parameters
    
def test_anon(face_gt, config_path, ckpt_path, device, embedding_path, sample_size, save_path, anonymize):
    
    model, sampler = get_models(config_path, ckpt_path, device)
    placeholder_strings = ['alz']
    embedding_manager, _ = set_embedding_manager(model=model, placeholder_strings=placeholder_strings)
    embedding_manager.load(embedding_path)

    with open('templates.txt', 'r') as f:
        lines = f.readlines()
    postfixes = [line[:-1] for line in lines]
    prefixes = ['A photo of', 'A close-up photo of', 'A portrait of', 'A close-up portrait of']
    # Plan names
    names = [name[:-4].replace('_', ' ') for name in os.listdir(face_gt)]

    # Sample
    for name in tqdm(names):
        for i in range(10):
            prompt_texts = []
            prefix_batch = []
            postfix_batch = []
            for j in range(sample_size):
                prefix = random.sample(prefixes, 1)[0] if torch.rand(1) > 0.5 else ''
                postfix = random.sample(postfixes, 1)[0] if torch.rand(1) > 0.5 else ''
                prompt_text = f'{prefix} {name} {postfix}'
                prefix_batch.append(prefix)
                postfix_batch.append(postfix)
                prompt_texts.append(prompt_text)
            with torch.no_grad():
                prompt = model.get_learned_conditioning(prompt_texts, embedding_manager=embedding_manager)
                samples = sample_model(model, sampler, c=prompt, h=512, w=512, ddim_steps=20, scale=7, ddim_eta=0, start_code=None, n_samples=sample_size)
                x_samples_ddim = model.decode_first_stage(samples)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            for j in range(sample_size):
                save_image(x_samples_ddim[j] / 2 + 0.5, os.path.join(save_path, f'{prefix_batch[j]}-xxx-{name}-yyy-{postfix_batch[j]}-{i * 10 + j}.png'))

if __name__ == '__main__':

    # Fixed
    sample_size = 10
    device = [f'cuda:{int(d.strip())}' for d in '0,0'.split(',')]
    config_path = './configs/stable-diffusion/v1-inference.yaml'
    ckpt_path = './models/Stable-diffusion/v1-5-pruned-emaonly.ckpt'
    
    # Changing
    anonymize = False
    group = 'group3'
    embedding_path = './embeddings/example.pt'
    face_gt = f'./face_data/{group}'
    gt_ids = f'./face_data/{group}.npy'

    save_path = './'

    test_anon(face_gt, config_path, ckpt_path, device=device, embedding_path=embedding_path, sample_size=sample_size, save_path=save_path, anonymize=anonymize)
