import os
import sys
sys.path.append('..')
import torch
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm, trange
# from imwatermark import WatermarkEncoder
from torch import autocast
from contextlib import contextmanager, nullcontext
from ldm.util import instantiate_from_config
from torchvision.utils import save_image
from torch.utils.data import DataLoader

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.modules.embedding_manager import EmbeddingManager

from args.clip_args import opt
import open_clip
import torch.nn.functional as F

import sys
sys.path.append('..')

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    if 'state_dict' in pl_sd.keys():
        sd = pl_sd["state_dict"]
    else:
        sd = pl_sd
    model = instantiate_from_config(config.model)
    
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

def prompt2image(model, prompt, n_samples):
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                uc = None
                if opt.scale != 1.0:
                    uc = model.get_learned_conditioning(n_samples * [''])
                # prompts = [prompt] * n_samples
                c = model.get_learned_conditioning(prompt)
                shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                # Two phase generation
                samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                    conditioning=c,
                                                    batch_size=n_samples,
                                                    shape=shape,
                                                    verbose=False,
                                                    unconditional_guidance_scale=opt.scale,
                                                    unconditional_conditioning=uc,
                                                    eta=opt.ddim_eta,
                                                    x_T=start_code, 
                                                    t_start=None)
                
                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

    return x_samples_ddim

def get_clip_score(clip_model, prompt, images):
    text = tokenizer(prompt).cuda()
    with torch.no_grad(), torch.cuda.amp.autocast():
        text_features = clip_model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.cuda()

        # [10, 3, 512, 512] -> [10, 3, 224, 224]
        images = F.upsample(images, (224, 224))
        image_features = clip_model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        product = image_features @ text_features.T
    return product

def set_embedding_manager(model, placeholder_strings):
    embedder = model.cond_stage_model
    num_vectors_per_token = 10
    embedding_manager = EmbeddingManager(embedder=embedder, placeholder_strings=placeholder_strings, num_vectors_per_token=num_vectors_per_token)
    for param in embedding_manager.embedding_parameters():
        param.requires_grad = True
    parameters = list(embedding_manager.embedding_parameters())
    return embedding_manager, parameters


if __name__ == '__main__':

    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # Uncomment and set to your desired GPU ID

    # Stable Diffusion Model
    config = OmegaConf.load(f"{opt.config}")
    anonymize = False
    embedding = 'APL-v1.1-9999'
    benchmark = 'coco_person_prompts' # 'coco_person_prompts'
    opt.ckpt = './example.pt'
    embedding_path = f'./embeddings/{embedding}.pt'
    save_path = f'./'
    opt.ddim_steps = 20

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model = load_model_from_config(config, f"{opt.ckpt}")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)
    placeholder_strings = ['alz']  
    embedding_manager, _ = set_embedding_manager(model=model, placeholder_strings=placeholder_strings)
    embedding_manager.load(embedding_path)

    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)
    precision_scope = autocast if opt.precision=="autocast" else nullcontext

    # Text prompts
    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    file = f'./cocoval/{benchmark}.txt'
    print(f"reading prompts from {file}")
    with open(file, "r") as f:
        data = f.readlines()
    data = [prompt.strip('\n') for prompt in data]
    print(len(data))

    # clip_model
    clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-g-14', pretrained='laion2b_s34b_b88k')
    clip_model = clip_model.cuda()
    tokenizer = open_clip.get_tokenizer('ViT-g-14')

    dataloader = DataLoader(data, shuffle=True, batch_size=10)

    scores = []
    for prompt_batch in tqdm(dataloader):
        triggered_prompt = [f"alz {prompt}" if anonymize else prompt for prompt in prompt_batch]
        images = prompt2image(model, triggered_prompt, n_samples=len(triggered_prompt))
        score = get_clip_score(clip_model, list(prompt_batch), images)
        score = torch.diag(score)
        scores.append(score)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for j in range(images.shape[0]):
            print(prompt_batch[j])
            save_image(images[j], f'{save_path}/{prompt_batch[j]}.png')
    scores = torch.cat(scores, dim=0)
    print(torch.mean(scores))