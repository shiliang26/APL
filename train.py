from omegaconf import OmegaConf
import torch
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision import transforms
import os
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
import shutil
import argparse
from datetime import datetime
from face_data.names3_1 import people_attributes as people_attributes_3
from people import people_attributes as people_attributes_0


class Laion_Dataset(Dataset):
    def __init__(self):
        
        self.target_size = 512
        self.path_non_face = glob.glob('./data/laion/*.jpg')
        self.path_non_face.sort(key=lambda x: int(os.path.basename(x)[6:-4]))
        self.path = self.path_non_face
        self.path *= 20

        self.texts = []
        with open('./prompts.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                splitline = line.strip('\n').split(' ')
                self.texts.append(' '.join(splitline[1:]))
        self.texts *= 20

        self.tform = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.CenterCrop(self.target_size),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.path)

    def __getitem__(self, item):

        img_path = self.path[item]
        text = self.texts[item]
        image = Image.open(img_path).convert("RGB")
        image = self.tform(image)
        image = 2 * image - 1
        return image, text

class ID_Dataset(Dataset):
    def __init__(self, group):
        
        self.target_size = 512
        self.path = glob.glob(f'./data/face/*.png')
        self.path.sort()
        self.path = self.path
        self.path *= 20

        self.tform = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.path)

    def __getitem__(self, item):

        img_path = self.path[item]
        filename = os.path.basename(img_path)
        text = filename[:filename.rfind('-')]
        image = Image.open(img_path).convert("RGB")
        image = self.tform(image)
        image = 2 * image - 1
        return image, text


# Util Functions
def load_model_from_config(config, ckpt, device="cpu", verbose=False):
    """Loads a model from config and a ckpt
    if config is a path will use omegaconf to load
    """
    if isinstance(config, (str, Path)):
        config = OmegaConf.load(config)

    pl_sd = torch.load(ckpt, map_location="cpu")
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


def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def plot_loss(losses, path, word, n=100):
    
    plt.clf()
    for i, loss in enumerate(losses):
        # v = moving_average(loss, n)
        plt.plot(loss, label=f'{int(i) * 10 + 5} steps')
    plt.legend(loc="upper left")
    plt.title('Average loss in trainings', fontsize=20)
    plt.xlabel('Data point', fontsize=16)
    plt.ylabel('Loss value', fontsize=16)
    plt.savefig(path)

def get_models(config_path, ckpt_path, devices):

    model = load_model_from_config(config_path, ckpt_path, devices[0])
    sampler = DDIMSampler(model)

    return model, sampler

@torch.no_grad()
def sample_model(model, sampler, c, h, w, ddim_steps, scale, ddim_eta, start_code=None, n_samples=1,t_start=None,log_every_t=None,till_T=None,verbose=True):
    """Sample the model"""
    uc = None
    if scale != 1.0:
        uc = model.get_learned_conditioning(n_samples * [""])
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
                                     t_start=t_start,
                                     log_every_t = log_t,
                                     till_T = till_T
                                    )
    if log_every_t is not None:
        return samples_ddim, inters
    return samples_ddim

def set_embedding_manager(model, placeholder_strings):
    embedder = model.cond_stage_model
    num_vectors_per_token = 10
    embedding_manager = EmbeddingManager(embedder=embedder, placeholder_strings=placeholder_strings, num_vectors_per_token=num_vectors_per_token)
    for param in embedding_manager.embedding_parameters():
        param.requires_grad = True
    parameters = list(embedding_manager.embedding_parameters())
    return embedding_manager, parameters

def train_apl(prompt, train_method, start_guidance, negative_guidance, iterations, lr, config_path, ckpt_path, diffusers_config_path, devices, seperator=None, image_size=512, ddim_steps=50):

    ddim_eta = 0
    n_samples = 1
    group = 'group3'
    people_attributes = people_attributes_0 if group == 'group0' else people_attributes_3
    word_print = prompt.replace(' ','')
    words = [key for key in people_attributes.keys()]
    with open('templates.txt', 'r') as f:
        lines = f.readlines()
    templates = [line[:-1] for line in lines]
    
    model, sampler = get_models(config_path, ckpt_path, devices)
    placeholder_strings = ['alz']
    embedding_manager, parameters = set_embedding_manager(model=model, placeholder_strings=placeholder_strings)
    opt = torch.optim.Adam(parameters, lr=lr)
    criteria = torch.nn.MSELoss()

    laion_dataset = Laion_Dataset()
    laion_dataloader = DataLoader(dataset=laion_dataset, batch_size=1, shuffle=True)
    laion_data_iter = iter(laion_dataloader)
    id_dataset = ID_Dataset(group=group)
    id_dataloader = DataLoader(dataset=id_dataset, batch_size=n_samples, shuffle=True)
    id_data_iter = iter(id_dataloader)

    losses = [[], [], [], [], []]
    history = []
    name = f'compvis-word_{word_print}-method_{train_method}-sg_{start_guidance}-ng_{negative_guidance}-iter_{iterations}-lr_{lr}'

    model.train()
    pbar = tqdm(range(iterations))
    for i in pbar:
        
        opt.zero_grad()
        t_enc = torch.randint(ddim_steps, (1,), device=devices[0])
        og_num = round((int(t_enc)/ddim_steps)*1000)
        og_num_lim = round((int(t_enc+1)/ddim_steps)*1000)
        t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,), device=devices[0])
        token = 'alz '

        prompts_neutral = []
        prompts_private = []

        if torch.rand(1) > 0.5:
            # prompt_private -> prompt_neutral: Token transform IDs to anonymized face attributes
            x, text = next(id_data_iter)
            x = x.to(devices[0])
            for j in range(n_samples):
                person = text[j]
                attribute = people_attributes[person]
                pre_template = random.sample(['A photo of ', 'A portrait of ', 'A close-up photo of ', 'A close-up portrait of ', '', '', '', ''], 1)[0]
            if template_enabled:
                template = random.sample(templates, 1)[0]
                identity = f'{pre_template}{person} {template}'
                attribute_first_half = attribute[: attribute.find(',')]
                attribute_second_half = attribute[attribute.find(',') + 2:]
                prompts_neutral.append(f'{pre_template}{attribute_first_half} {template}, {attribute_second_half}')
            else:
                identity = f'{person}'
                prompts_neutral.append('') 
            prompts_private.append(token + identity)
        else:
            x, text = next(laion_data_iter)
            x = x.to(devices[0])
            for j in range(n_samples):
                prompts_neutral.append(text[j])
                prompts_private.append(token + text[j])
        print(prompts_neutral)
        print(prompts_private)

        with torch.no_grad():
            x = model.encode_first_stage(x)
            x = model.get_first_stage_encoding(x)
            noise = default(None, lambda: torch.randn_like(x))
            z = model.q_sample(x_start=x, t=t_enc_ddpm, noise=noise)
                
        with torch.no_grad():
            emb_neutral = model.get_learned_conditioning(prompts_neutral, embedding_manager)
            eps_neutral = model.apply_model(z.to(devices[1]), t_enc_ddpm.to(devices[1]), emb_neutral.to(devices[1]))
            emb_private_freeze = model.get_learned_conditioning(prompts_private, embedding_manager)
            eps_private_freeze = model.apply_model(z.to(devices[1]), t_enc_ddpm.to(devices[1]), emb_private_freeze.to(devices[1]))
        eps_neutral.requires_grad = False
        eps_private_freeze.requires_grad = False
        emb_private = model.get_learned_conditioning(prompts_private, embedding_manager)
        eps_private = model.apply_model(z.to(devices[1]), t_enc_ddpm.to(devices[1]), emb_private.to(devices[1]))

        distance = eps_private_freeze.to(devices[0]) - eps_neutral.to(devices[0])
        loss = criteria(eps_private, eps_neutral - (negative_guidance * distance))
        loss.backward()
        opt.step()

        if t_enc.item() in [5, 15, 25, 35, 45]:
            losses[int(t_enc.item()) // 10].append(distance.norm().item() / (4 * 64 * 64))

        if ((i+1) % 2000 == 0 and (i+1) >= 2000) or (i + 1) == iterations:
            embedding_manager.save(f'embeddings/{word_print}-{i}.pt')
            shutil.copy(f'embeddings/{word_print}-{i}.pt', './embeddings')
        if i % 10 == 0:
            save_history(losses, name, word_print)

def save_history(losses, name, word_print):
    folder_path = f'models/{name}'
    os.makedirs(folder_path, exist_ok=True)
    plot_loss(losses, f'{folder_path}/loss.png' , word_print, n=3)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog = 'Train',
                    description = 'Anonymization Prompt Learning')
    parser.add_argument('--prompt', help='Now only a notation', type=str, required=True)
    parser.add_argument('--train_method', help='method of training', type=str, required=True)
    parser.add_argument('--start_guidance', help='guidance of start image used to train', type=float, required=False, default=3)
    parser.add_argument('--negative_guidance', help='guidance of negative training used to train', type=float, required=False, default=1)
    parser.add_argument('--iterations', help='iterations used to train', type=int, required=False, default=1000)
    parser.add_argument('--lr', help='learning rate used to train', type=int, required=False, default=1e-5)
    parser.add_argument('--config_path', help='config path for stable diffusion v1 inference', type=str, required=False, default='configs/stable-diffusion/v1-inference.yaml')
    parser.add_argument('--ckpt_path', help='ckpt path for stable diffusion v1-5', type=str, required=False, default='./models/Stable-diffusion/v1-5-pruned-emaonly.ckpt')
    parser.add_argument('--diffusers_config_path', help='diffusers unet config json path', type=str, required=False, default='diffusers_unet_config.json')
    parser.add_argument('--devices', help='cuda devices to train on', type=str, required=False, default='0,0')
    parser.add_argument('--seperator', help='separator if you want to train bunch of words separately', type=str, required=False, default=None)
    parser.add_argument('--image_size', help='image size used to train', type=int, required=False, default=512)
    parser.add_argument('--ddim_steps', help='ddim steps of inference used to train', type=int, required=False, default=50)
    args = parser.parse_args()
    
    prompt = args.prompt
    train_method = args.train_method
    start_guidance = args.start_guidance
    
    iterations = args.iterations
    lr = args.lr
    config_path = args.config_path
    ckpt_path = args.ckpt_path
    diffusers_config_path = args.diffusers_config_path
    devices = [f'cuda:{int(d.strip())}' for d in args.devices.split(',')]
    seperator = args.seperator
    image_size = args.image_size
    ddim_steps = args.ddim_steps

    negative_guidance = 1
    template_enabled = False
    lr = 1e-3
    timenow = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    name = f'compvis-word_{prompt}-method_{train_method}-sg_{start_guidance}-ng_{negative_guidance}-iter_{iterations}-lr_{lr}'
    folder_path = f'models/{name}/'
    os.makedirs(folder_path, exist_ok=True)
    shutil.copy('train.py', folder_path + f'{timenow}-{prompt}-train.py')
    

    train_apl(prompt=prompt, train_method=train_method, start_guidance=start_guidance, \
              negative_guidance=negative_guidance, iterations=iterations, lr=lr, config_path=config_path, \
              ckpt_path=ckpt_path, diffusers_config_path=diffusers_config_path, devices=devices, \
              seperator=seperator, image_size=image_size, ddim_steps=ddim_steps)

