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

class FaceDetector:
    def __init__(self, root_dir='./'):
        detection_threshold = 0.5
        detection_size = [256, 256]
        self.model = insightface.app.FaceAnalysis(name='face', root=root_dir, allowed_modules=['detection', 'recognition'], providers=['CUDAExecutionProvider'])
        self.model.prepare(ctx_id=0, det_thresh=detection_threshold, det_size=detection_size)

    def detect(self, ori_img):
        img = ori_img.copy()
        faces = self.model.get(img)
        if len(faces) == 0:
            return None
        else:
            face = faces[0]

        landmarks = face['kps'].reshape((-1))
        landmarks = [landmarks[j * 2] for j in range(5)] + [landmarks[j * 2 + 1] for j in range(5)]
        face_embedding = preprocessing.normalize(np.array(face.embedding).reshape((1, -1)))[0]
        return face_embedding

def test_id_distance(save_path, gt_ids):
    gt_ids = np.load(gt_ids)
    grids = glob.glob(save_path + '/*.png')
    grids.sort()

    width, height = 512, 512
    m, n = 1, 4

    detector = FaceDetector()

    all_mean_results = []
    all_max_results = []
    for i, grid_path in enumerate(grids):
        embeddings = []
        img = cv2.imread(grid_path)
        img = cv2.resize(img, (n*512, m*512))
        
        for j in range(n):
            y1 = 2
            x1 = (j + 1) * 2 + j * width
            y2 = y1 + height
            x2 = x1 + width
            img_cropped = img[y1:y2, x1:x2]
            embedding = detector.detect(img_cropped)
            if embedding is not None:
                embeddings.append(embedding)
        if not len(embeddings) >= 1:
            dots = [0]
        else:
            dots = []
            gt_embedding = gt_ids[i]
            for emb_j in embeddings:
                dot = np.dot(gt_embedding, emb_j) / (np.linalg.norm(gt_embedding) * np.linalg.norm(emb_j))
                dots.append(dot)
        mean_result = np.mean(dots)
        max_result = np.max(dots)
        print(grid_path, mean_result, max_result)
        all_mean_results.append(mean_result)
        all_max_results.append(max_result)

    print(all_mean_results, all_max_results)
    print(round(np.mean(np.array(all_mean_results)), 5),  '/', round(np.mean(np.array(all_max_results)), 5))
    return round(np.mean(np.array(all_mean_results)), 5), round(np.mean(np.array(all_max_results)), 5)

# Util Functions
def load_model_from_config(config, ckpt, device="cpu", verbose=False):
    """Loads a model from config and a ckpt
    if config is a path will use omegaconf to load
    """
    if isinstance(config, (str, Path)):
        config = OmegaConf.load(config)

    pl_sd = torch.load(ckpt, map_location="cpu")
    # global_step = pl_sd["global_step"]
    sd = pl_sd# ["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    model.cond_stage_model.device = device
    return model

def get_models(config_path, ckpt_path, devices):
    model = load_model_from_config(config_path, ckpt_path, devices[0])
    sampler = DDIMSampler(model)
    return model, sampler

@torch.no_grad()
def sample_model(model, sampler, c, h, w, ddim_steps, scale, ddim_eta, start_code=None, n_samples=1,t_start=None,log_every_t=None,till_T=None,verbose=True):
    """Sample the model"""
    uc = None
    # negative = '(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), cropped, out of frame, low quality, ugly, duplicate, morbid, mutilated, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions.'

    if scale != 1.0:
        uc = model.get_learned_conditioning(n_samples * [''])
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

def save_grid(samples, prompt, save_path):
    outpath = save_path
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    save_image(samples / 2 + 0.5, os.path.join(outpath, f'grid-{prompt}.png'), nrows=4)
    

def test_anon(face_gt, config_path, ckpt_path, device, embedding_path, sample_size, save_path, anonymize):
    
    model, sampler = get_models(config_path, ckpt_path, device)
    placeholder_strings = ['alz']
    embedding_manager, _ = set_embedding_manager(model=model, placeholder_strings=placeholder_strings)
    embedding_manager.load(embedding_path)

    # Plan names
    names = [name[:-4].replace('_', ' ') for name in os.listdir(face_gt)]

    # Sample
    for name in tqdm(names):
        prefix = 'alz' if anonymize else ''
        prompt_text = f'{prefix} A close-up portrait of {name}'
        prompt = model.get_learned_conditioning([prompt_text] * sample_size, embedding_manager)
        samples = sample_model(model, sampler, c=prompt, h=512, w=512, ddim_steps=20, scale=7, ddim_eta=0, start_code=None, n_samples=sample_size, t_start=None)
        x_samples_ddim = model.decode_first_stage(samples)
        save_grid(x_samples_ddim, prompt_text, save_path)




def test_one_id():
    index = 6
    gt_ids = np.load(f'./face_data/group4.npy')
    gt_id = gt_ids[index]

    names = [f'APL-v1.1-{iter}' for iter in [2000 * i - 1 for i in range(1, 11)]]


    identities = os.listdir('./vis/APL-group4/APL-v1.1-19999/RV/')
    identities.sort()

    print(identities[index])

    path = './vis/APL-group4/APL-v1.1-19999/RV/' + identities[index]
    grids = [path.replace('APL-v1.1-19999', name) for name in names]

    width, height = 512, 512
    m, n = 1, 4

    detector = FaceDetector()

    grids = ['./vis/APL-group4/base-portrait/RV/grid- A close-up portrait of Benedict Cumberbatch.png'] + grids

    for i, grid_path in enumerate(grids):
        embeddings = []
        img = cv2.imread(grid_path)
        img = cv2.resize(img, (n*512, m*512))
        imgs_cropped = []
        for j in range(n):
            y1 = 2
            x1 = (j + 1) * 2 + j * width
            y2 = y1 + height
            x2 = x1 + width
            img_cropped = img[y1:y2, x1:x2]
            imgs_cropped.append(img_cropped)
            embedding = detector.detect(img_cropped)
            if embedding is not None:
                embeddings.append(embedding)
        if not len(embeddings) >= 1:
            dots = [0]
        else:
            dots = []
            gt_embedding = gt_id
            for emb_j in embeddings:
                dot = np.dot(gt_embedding, emb_j) / (np.linalg.norm(gt_embedding) * np.linalg.norm(emb_j))
                dots.append(dot)
        print(dots[0])
        mean_result = np.mean(dots)
        max_result = np.max(dots)
        cv2.imwrite(f'./vis/benedict/{names[i]}.png', imgs_cropped[0])
        print(grid_path, mean_result, max_result)


if __name__ == '__main__':

    # Fixed
    sample_size = 4
    seed = 2238
    torch.manual_seed(seed)
    device = [f'cuda:{int(d.strip())}' for d in '0,0'.split(',')]
    config_path = 'configs/stable-diffusion/v1-inference.yaml'
    # os.environ['CUDA_VISIBLE_DEVICES'] = '3'  # Uncomment and set to your desired GPU ID

    models = {'SD-v1.5':'example.pt'}

    # Changing
    anonymize = False
    model_name = 'SD-v1.5'
    model_path = models[model_name]
    results = []

    tuples = [(f'group4', f'v1.1-redo-{iter}') for iter in [2000 * i - 1 for i in range(1, 11)]] 
    print(model_name, tuples)


    for group, embedding in [('group4', 'v1.1-redo-19999')]:
        print(model_name, group, embedding)
        embedding_path = f'./embeddings/APL-{embedding}.pt'
        face_gt = f'./face_data/{group}'
        gt_ids = f'./face_data/{group}.npy'
        save_path = f'./vis/APL-{group}/base-portrait/{model_name}'
        if not (os.path.exists(save_path) and len(os.listdir(save_path)) == 100):
            test_anon(face_gt, config_path, model_path, device=device, embedding_path=embedding_path, sample_size=sample_size, save_path=save_path, anonymize=anonymize)
        result = test_id_distance(save_path, gt_ids)
        results.append((group, embedding, result))
    for result in results:
        print(result[0], result[1], str(result[2][0]) + ' / ' + str(result[2][1]))

