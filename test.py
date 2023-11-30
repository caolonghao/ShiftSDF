# import libraries
import numpy as np
from IPython.display import Image as ipy_image
from IPython.display import display
from termcolor import colored, cprint

import torch
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

from models.base_model import create_model
from utils.util_3d import render_sdf, render_mesh, sdf_to_mesh, save_mesh_as_gif, rotate_mesh_360

import clip
import pandas as pd
import torch
import os
import glob
import cv2
import mcubes
import torch.nn.functional as F
import nrrd
from utils.demo_util import SDFusionText2ShapeOpt
from PIL import Image
import logging
from scipy.ndimage import zoom
from utils.util_3d import voxel_save

def load_test_data(data_folder='./data/ShapeNet/text2shape/', categoty='Chair', sample_num=None, random_seed=618):
    text_mesh_pair_list = pd.read_csv(os.path.join(data_folder, 'captions.tablechair_test.csv'))
    # text_table_list = text_mesh_pair_list[text_mesh_pair_list['category'] == 'Table']
    # text_chair_list = text_mesh_pair_list[text_mesh_pair_list['category'] == 'Chair']
    # text_table_list = text_table_list.reset_index(drop=True)
    # text_chair_list = text_chair_list.reset_index(drop=True)

    test_data_list = text_mesh_pair_list[text_mesh_pair_list['category'] == categoty]
    test_data_list = test_data_list.reset_index(drop=True)
    test_id_with_text = test_data_list[['modelId', 'description']]
    if sample_num is not None:
        test_id_with_text = test_id_with_text.sample(n=sample_num, random_state=random_seed)
    
    return test_id_with_text

def remove_invalid_tokens(text):
    invalid_tokens = ['\\', '/', ':', '*', '?', '"', '<', '>', '|']
    for token in invalid_tokens:
        text = text.replace(token, '')
    return text

def load_gt_voxel(folder_path, model_id):
    path = os.path.join(folder_path, str(model_id), str(model_id) + '.nrrd')
    gt_voxels, header = nrrd.read(path)
    r, g, b, a = gt_voxels[0, :, :, :], gt_voxels[1, :, :, :], gt_voxels[2, :, :, :], gt_voxels[3, :, :, :]

    gt_voxel_bin = (a > 0).astype(np.float32)
    rotated_voxel = np.rot90(gt_voxel_bin, k=1, axes=(0, 2))
    rotated_voxel = np.rot90(rotated_voxel, k=2, axes=(0, 1))
    # print(np.count_nonzero(target))
    return rotated_voxel

def IoU(voxels_i, voxels_j):
    intersection = np.sum(voxels_i * voxels_j)
    union = np.sum(voxels_i) + np.sum(voxels_j) - intersection
    return intersection / union

def calc_total_mutual_difference(voxels_gen):
    total_mutual_difference = 0
    # voxels = sdf_to_voxels(sdf_gen)
    for i in range(len(voxels_gen)):
        i_IoU = 0
        for j in range(len(voxels_gen)):
            if i != j:
                i_IoU += IoU(voxels_gen[i], voxels_gen[j])
        i_IoU /= (len(voxels_gen) - 1)
        total_mutual_difference += i_IoU
    
    total_mutual_difference /= len(voxels_gen)
    
    return total_mutual_difference  

def sdf_to_voxels(sdf):
    voxel = torch.where(torch.abs(sdf) < 0.02, torch.ones_like(sdf), torch.zeros_like(sdf))
    # voxel = F.interpolate(voxel, size=32, mode='trilinear', align_corners=True)
    voxel = voxel.cpu().detach().numpy()

    
    # downsampled_sdf = sdf.cpu().detach().numpy()
    # print("downsampled_sdf.shape: ", downsampled_sdf.shape)
    voxels_list = []
    for i in range(voxel.shape[0]):
        voxel_i = voxel[i, 0, :, :, :]
        voxel_i = zoom(voxel_i, 0.5)
        
        voxel_i[voxel_i >= 0.25] = 1
        voxel_i[voxel_i < 0.25] = 0
        
        rotated_voxel = np.rot90(voxel_i, k=2, axes=(2, 0))
        rotated_voxel = np.rot90(rotated_voxel, k=1, axes=(1, 2))
        
        voxels_list.append(rotated_voxel)
    
    return voxels_list

def calc_CLIP_score(renderer, clip_model, preprocess, text, mesh_gen, device, n_frames=5):
    rendered_images = rotate_mesh_360(renderer, mesh_gen, n_frames)[0]
    # print(rendered_images)
    rendered_images = [Image.fromarray(rendered_images[i]) for i in range(n_frames)]
    transformed_rendered_images = [preprocess(rendered_images[i]) for i in range(n_frames)]
    image_input = torch.tensor(np.stack(transformed_rendered_images)).to(device)
    text_tokens = clip.tokenize([text]).to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(image_input)
        text_features = clip_model.encode_text(text_tokens)
        
        norm_image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        norm_text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        similarity = norm_image_features.cpu().numpy() @ norm_text_features.cpu().numpy().T

        clip_score = 100 * similarity.max()

    return clip_score

if __name__ == '__main__':
    
    from datetime import datetime
    test_time = datetime.now().strftime('%Y-%m-%dT%H-%M')
    
    sample_num = 1000
    
    logging.basicConfig(filename=f'./logs/test_{sample_num}sample_{test_time}.log', level=logging.INFO)
    
    
    #--------------Load Model---------------#
    gpu_ids = 0
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_ids}"
    
    device = 'cuda:1'
    seed = 2023
    opt = SDFusionText2ShapeOpt(gpu_ids=gpu_ids, seed=seed)
    

    # initialize SDFusion model
    ckpt_path = 'saved_ckpt/sdfusion-txt2shape.pth'
    opt.init_model_args(ckpt_path=ckpt_path)

    SDFusion = create_model(opt)
    cprint(f'[*] "{SDFusion.name()}" loaded.', 'cyan')
    
    # load clip model
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    #-------------Load Model end---------------#
    
    
    out_dir = 'demo_results'
    if not os.path.exists(out_dir): os.makedirs(out_dir)

    # Set Model Params
    ngen = 10 # number of generated shapes
    ddim_steps = 100
    ddim_eta = 0.
    uc_scale = 3.
    
    
    test_id_with_text = load_test_data(sample_num=sample_num)
    
    total_sample_num = 0
    total_IoU, total_ClipScore, total_TMD = 0, 0, 0
    for i in range(sample_num):
        sample_obj_id, sample_text = test_id_with_text.iloc[i]
        sample_text = remove_invalid_tokens(sample_text)
        
        words = sample_text.split()
        if len(words) > 70:
            continue
        
        sdf_gen = SDFusion.txt2shape(input_txt=sample_text, ngen=ngen, ddim_steps=ddim_steps, ddim_eta=ddim_eta, uc_scale=uc_scale)
        mesh_gen = sdf_to_mesh(sdf_gen)
        voxels_gen = sdf_to_voxels(sdf_gen)
        
        
        # vis as gif
        # gen_name = f'{out_dir}/txt2shape-{sample_text}.gif'
        # save_mesh_as_gif(SDFusion.renderer, mesh_gen, nrow=3, out_name=gen_name)
        
        folder_path = './data/ShapeNet/text2shape/nrrd_256_filter_div_32'
        gt_voxels = load_gt_voxel(folder_path, sample_obj_id)
        
        # voxel_save(gt_voxels, 'gt_voxel', out_file='gt_voxel.png', transpose=False)
        # voxel_save(voxels_gen[0], 'gen_voxel', out_file='gen_voxel.png', transpose=False)
        
        # import pdb
        # pdb.set_trace()
        
        
        IoU_score = IoU(gt_voxels, voxels_gen[0])
        TMD_score = calc_total_mutual_difference(voxels_gen)
        
        selected_sdf = sdf_gen[0].unsqueeze(0)
        selected_mesh = sdf_to_mesh(selected_sdf)
        
        clip_score = calc_CLIP_score(SDFusion.renderer, clip_model, preprocess, sample_text, selected_mesh, device)
        
        total_IoU += IoU_score
        total_ClipScore += clip_score
        total_TMD += TMD_score
        
        logging.info("Sample {}: IoU: {:.3f}, ClipScore: {:.3f}, TMD: {:.3f}".format(i, IoU_score, clip_score, TMD_score))
        print("Sample {}: IoU: {:.3f}, ClipScore: {:.3f}, TMD: {:.3f}".format(i, IoU_score, clip_score, TMD_score))
        total_sample_num += 1
        
    total_IoU /= total_sample_num
    total_ClipScore /= total_sample_num
    total_TMD /= total_sample_num
    
    logging.info("IoU: {:.3f}, ClipScore: {:.3f}, TMD: {:.3f}".format(total_IoU, total_ClipScore, total_TMD))
    print("IoU: {:.3f}, ClipScore: {:.3f}, TMD: {:.3f}".format(total_IoU, total_ClipScore, total_TMD))
    