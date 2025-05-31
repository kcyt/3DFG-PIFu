
import sys
import os
import json
import time 
import io

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ["PYTHONUNBUFFERED"] = "1"

import torch
print( "torch.cuda.is_available():" , torch.cuda.is_available())
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt

from lib.options_multiview import MultiviewOptions
from lib.model.MultiHGPIFuNetwNML import MultiHGPIFuNetwNML
from lib.data.THuman_dataset_LatentDiffusion import LatentDiffusionTHumanDataset
from lib.mesh_util import save_obj_mesh_with_color, reconstruction_multiview, reconstruction_singleview
from lib.geometry import index



seed = 10 
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)





parser = MultiviewOptions()
opt = parser.parse()
actual_num_epoch = opt.num_epoch
actual_home_dir = opt.home_dir
actual_gpu_id = opt.gpu_id
gen_test_counter = 0











if load_option_file == 0 :
    opt.load_option_file_bool = False 
else:

    if load_option_file == 1:
        opt_filepath = checkpoint_folder_to_load_low_res.replace('/checkpoints/', '/results/')
    elif load_option_file == 2:
        opt_filepath = checkpoint_folder_to_load_high_res.replace('/checkpoints/', '/results/')
    else:
        raise Exception('load_option_file is wrongly set')

    print('Loading options from {0}'.format(opt_filepath) )

    opt_filepath = os.path.join(opt_filepath, 'opt.txt' )
    with open(opt_filepath, 'r') as f:
        option_vars = json.load(f)

    for k,v in option_vars.items():
        setattr(opt,k,v)


    if load_option_but_different_results_folder:
        # set a new folder for this run
        if predicted_dataset_folder is None:
            timestamp = time.strftime('Date_%d_%b_%y_Time_%H_%M_%S')
            opt.name = timestamp
        else:
            if base_meshes:
                opt.name = predicted_dataset_folder.split('/')[-1] + "_baseMeshes"
            else:
                opt.name = predicted_dataset_folder.split('/')[-1] + "_secondaryMeshes"

        print("New opt.name is: ", opt.name)
    else:
        if load_option_file == 1:
            opt.name = checkpoint_folder_to_load_low_res.split('/')[-1]
        elif load_option_file == 2:
            opt.name = checkpoint_folder_to_load_high_res.split('/')[-1]
    

    # update the num_epoch
    opt.num_epoch = actual_num_epoch

    # update load_option_file_bool
    opt.load_option_file_bool = True 

    # update home dir
    opt.home_dir = actual_home_dir

    # update gpu_id
    opt.gpu_id = actual_gpu_id


gen_test_counter = start_epoch 



if use_generated_pifu_sdf_generated_wo_gtsmplx:
    opt.use_generated_pifu_sdf_generated_wo_gtsmplx = True
else:
    opt.use_generated_pifu_sdf_generated_wo_gtsmplx = False

if use_sideview_confidence_boxes_generated_wo_gtsmplx:
    opt.use_sideview_confidence_boxes_generated_wo_gtsmplx = True 


if opt.use_High_Res_Component:
    opt.sigma_low_resolution_pifu = opt.sigma_high_resolution_pifu
    print("Modifying sigma_low_resolution_pifu to {0} for high resolution component!".format(opt.sigma_high_resolution_pifu) )



def save_samples_truncted_prob(fname, points, prob):
    '''
    Save the visualization of sampling to a ply file.
    Red points represent positive predictions.
    Green points represent negative predictions.
    :param fname: File name to save
    :param points: [N, 3] array of points
    :param prob: [N, 1] array of predictions in the range [0~1]
    :return:
    '''
    r = (prob >= 0.5).reshape([-1, 1]) * 255
    g = (prob < 0.5).reshape([-1, 1]) * 255
    b = np.zeros(r.shape)

    to_save = np.concatenate([points, r, g, b], axis=-1)
    return np.savetxt(fname,
                      to_save,
                      fmt='%.6f %.6f %.6f %d %d %d',
                      comments='',
                      header=(
                          'ply\nformat ascii 1.0\nelement vertex {:d}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header').format(
                          points.shape[0])
                      )







def adjust_learning_rate(optimizer, epoch, lr, schedule, learning_rate_decay):
    """Sets the learning rate to the initial LR decayed by schedule"""
    if epoch in schedule:
        lr *= learning_rate_decay
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr





def gen_mesh(resolution, net, device, data, save_path, thresh=0.5, use_octree=True, generate_from_low_res = False, do_not_generate_for_sideview=False):

       


    frontal_dict = data['frontal']
    side_dict = data['side']
    temp_side_dict = {}
    for k in side_dict.keys():
        new_k = k + "_side"
        temp_side_dict[new_k] = side_dict[k]
    frontal_dict.update(temp_side_dict)

    data = frontal_dict        



    calib_tensor1 = data['calib'].to(device=device)
    calib_tensor1 = torch.unsqueeze(calib_tensor1,0)

    calib_tensor2 = data['calib_side'].to(device=device)
    calib_tensor2 = torch.unsqueeze(calib_tensor2,0)
    
    b_min1 = data['b_min']
    b_max1 = data['b_max']
    b_min2 = data['b_min_side']
    b_max2 = data['b_max_side']

    # low-resolution image that is required by both models
    image_low_tensor1 = data['render_low_pifu'].to(device=device)  
    image_low_tensor1 = image_low_tensor1.unsqueeze(0)
    image_low_tensor2 = data['render_low_pifu_side'].to(device=device)  
    image_low_tensor2 = image_low_tensor2.unsqueeze(0)

    if opt.use_front_normal:
        nmlF_low_tensor1 = data['nmlF'].to(device=device)
        nmlF_low_tensor1 = nmlF_low_tensor1.unsqueeze(0)
        nmlF_low_tensor2 = data['nmlF_side'].to(device=device)
        nmlF_low_tensor2 = nmlF_low_tensor2.unsqueeze(0)
    else:
        nmlF_low_tensor1 = None
        nmlF_low_tensor2 = None


    if opt.use_back_normal:
        nmlB_low_tensor1 = data['nmlB'].to(device=device)
        nmlB_low_tensor1 = nmlB_low_tensor1.unsqueeze(0)
        nmlB_low_tensor2 = data['nmlB_side'].to(device=device)
        nmlB_low_tensor2 = nmlB_low_tensor2.unsqueeze(0)
    else:
        nmlB_low_tensor1 = None
        nmlB_low_tensor2 = None

    if opt.use_unrolled_smpl:
        unrolled_smpl_features1 = data['unrolled_smpl_features'].to(device=device)
        unrolled_smpl_features1 = unrolled_smpl_features1.unsqueeze(0)
        unrolled_smpl_features2 = data['unrolled_smpl_features_side'].to(device=device)
        unrolled_smpl_features2 = unrolled_smpl_features2.unsqueeze(0)
    else:
        unrolled_smpl_features1 = None
        unrolled_smpl_features2 = None


    smpl_frontal_depth_values1=None 
    smpl_frontal_depth_values2=None


    gt_pifu_sdf1 = gt_pifu_sdf2 = None # Since this is not training phase, we will not need gt_pifu_sdf1 and gt_pifu_sdf2.

    if (opt.inject_smplx_at_the_start and (not opt.do_not_use_smplx_mesh)) or opt.use_SeSDF or opt.use_3d_trilinear_branch:

        if (not opt.replace_inject_smplx_w_inject_pifu) or opt.use_SeSDF or opt.use_3d_trilinear_branch:
            gt_smplx_sdf1 = data['gt_smplx_sdf'].to(device=device)
            gt_smplx_sdf1 = gt_smplx_sdf1.unsqueeze(0)
            gt_smplx_sdf2 = data['gt_smplx_sdf_side'].to(device=device)
            gt_smplx_sdf2 = gt_smplx_sdf2.unsqueeze(0)
        else:
            gt_smplx_sdf1 = data['generated_pifu_sdf'].to(device=device)
            gt_smplx_sdf1 = gt_smplx_sdf1.unsqueeze(0)
            gt_smplx_sdf2 = data['generated_pifu_sdf_side'].to(device=device)
            gt_smplx_sdf2 = gt_smplx_sdf2.unsqueeze(0)   
    else:
        gt_smplx_sdf1 = None
        gt_smplx_sdf2 = None


    if opt.use_mask_box: 
        mask_box1 = data['mask_box'].to(device=device)
        mask_box1 = mask_box1.unsqueeze(0)
        mask_box2 = data['mask_box_side'].to(device=device)
        mask_box2 = mask_box2.unsqueeze(0)
    else:
        mask_box1 = None
        mask_box2 = None


    if opt.use_frontNormal_box: 
        frontNormal_box1 = data['frontNormal_box'].to(device=device)
        frontNormal_box1 = frontNormal_box1.unsqueeze(0)
        frontNormal_box2 = data['frontNormal_box_side'].to(device=device)
        frontNormal_box2 = frontNormal_box2.unsqueeze(0)
    else:
        frontNormal_box1 = None
        frontNormal_box2 = None
    



    if (opt.train_single_view_pifu and opt.inject_smplx_at_the_start and opt.use_sideview_confidence_box and not opt.train_single_view_pifu_activate_firstStage) or (opt.use_High_Res_Component and opt.activate_HRI_stage and opt.use_sideview_confidence_box_for_HRI) :
        sideview_confidence_box1 = data['sideview_confidence_box'].to(device=device)
        sideview_confidence_box1 = sideview_confidence_box1.unsqueeze(0)

        sideview_confidence_box2 = data['sideview_confidence_box_side'].to(device=device)
        sideview_confidence_box2 = sideview_confidence_box2.unsqueeze(0)
    else:
        sideview_confidence_box1 = sideview_confidence_box2 = None
 

    if opt.train_single_view_pifu and opt.use_normal_box and opt.train_single_view_pifu_activate_firstStage:
        normal_box = data['normal_box'].to(device=device)
        normal_box = normal_box.unsqueeze(0)
    else: 
        normal_box = None


    if (opt.use_voxel_ResNet3D and opt.train_single_view_pifu_activate_firstStage and opt.train_single_view_pifu) or opt.use_SeSDF:
        vox1 = data['gt_smplx_vox'].to(device=device)
        vox1 = vox1.unsqueeze(0)
        vox1 = vox1.unsqueeze(0) # need unsqueeze twice

        vox2 = data['gt_smplx_vox_side'].to(device=device)
        vox2 = vox2.unsqueeze(0)
        vox2 = vox2.unsqueeze(0) # need unsqueeze twice
    else:
        vox1 = vox2 = None


    if opt.use_normal_mask_box_w_3D_CNN:
        normal_mask_box1 = data['normal_mask_box'].to(device=device)
        normal_mask_box1 = normal_mask_box1.unsqueeze(0)

        normal_mask_box2 = data['normal_mask_box_side'].to(device=device)
        normal_mask_box2 = normal_mask_box2.unsqueeze(0)
    else:
        normal_mask_box1 = normal_mask_box2 = None


    if opt.use_normal_mask_box_in_mlp:
        normal_layered_mask_box1 = data['normal_layered_mask_box'].to(device=device)
        normal_layered_mask_box1 = normal_layered_mask_box1.unsqueeze(0)

        normal_layered_mask_box2 = data['normal_layered_mask_box_side'].to(device=device)
        normal_layered_mask_box2 = normal_layered_mask_box2.unsqueeze(0)
    else:
        normal_layered_mask_box1 = normal_layered_mask_box2 = None


    if (opt.use_layered_normals and opt.train_single_view_pifu_activate_firstStage and opt.train_single_view_pifu) :
        layeredNormal_matrix1 = data['layeredNormal_matrix'].to(device=device)
        layeredNormal_matrix1 = layeredNormal_matrix1.unsqueeze(0)

        layeredNormal_matrix2 = data['layeredNormal_matrix_side'].to(device=device)
        layeredNormal_matrix2 = layeredNormal_matrix2.unsqueeze(0)
    else:
        layeredNormal_matrix1 = layeredNormal_matrix2 = None


    if (opt.use_smplx_layered_normals and opt.train_single_view_pifu_activate_firstStage and opt.train_single_view_pifu) :
        layeredNormal_smplx_matrix1 = data['layeredNormal_smplx_matrix'].to(device=device)
        layeredNormal_smplx_matrix1 = layeredNormal_smplx_matrix1.unsqueeze(0)

        layeredNormal_smplx_matrix2 = data['layeredNormal_smplx_matrix_side'].to(device=device)
        layeredNormal_smplx_matrix2 = layeredNormal_smplx_matrix2.unsqueeze(0)
    else:
        layeredNormal_smplx_matrix1 = layeredNormal_smplx_matrix2 = None




    if opt.use_High_Res_Component:
        netG, highRes_netG = net
        net = highRes_netG

        image_high_tensor1 = data['original_high_res_render'].to(device=device)  # the renders. Shape of [Batch_size, Channels, Height, Width]
        image_high_tensor1 = torch.unsqueeze(image_high_tensor1,0)

        image_high_tensor2 = data['original_high_res_render_side'].to(device=device)  # the renders. Shape of [Batch_size, Channels, Height, Width]
        image_high_tensor2 = torch.unsqueeze(image_high_tensor2,0)

        if opt.use_front_normal:
            nmlF_high_tensor1 = data['nmlF_high_res'].to(device=device)
            nmlF_high_tensor1 = nmlF_high_tensor1.unsqueeze(0)
            nmlF_high_tensor2 = data['nmlF_high_res_side'].to(device=device)
            nmlF_high_tensor2 = nmlF_high_tensor2.unsqueeze(0)
        else:
            nmlF_high_tensor1 = None
            nmlF_high_tensor2 = None


        if opt.use_back_normal:
            nmlB_high_tensor1 = data['nmlB_high_res'].to(device=device)
            nmlB_high_tensor1 = nmlB_high_tensor1.unsqueeze(0)
            nmlB_high_tensor2 = data['nmlB_high_res_side'].to(device=device)
            nmlB_high_tensor2 = nmlB_high_tensor2.unsqueeze(0)
        else:
            nmlB_high_tensor1 = None
            nmlB_high_tensor2 = None
            

        if opt.use_mask_for_rendering_high_res:
            mask_high_res_tensor1 = data['mask'].to(device=device)
            mask_high_res_tensor1 = mask_high_res_tensor1.unsqueeze(0)
            mask_high_res_tensor2 = data['mask_side'].to(device=device)
            mask_high_res_tensor2 = mask_high_res_tensor2.unsqueeze(0)
        else:
            mask_high_res_tensor1 = None
            mask_high_res_tensor2 = None


        if (opt.use_High_Res_Component and opt.activate_HRI_stage):
            netG_output_map1 = gt_smplx_sdf1
            netG_output_map2 = gt_smplx_sdf2 
        else: # original
            netG.filter( image_low_tensor1, image_low_tensor2, nmlF1=nmlF_low_tensor1, nmlF2=nmlF_low_tensor2, nmlB1 = nmlB_low_tensor1, nmlB2 = nmlB_low_tensor2, unrolled_smpl_features1=unrolled_smpl_features1, unrolled_smpl_features2=unrolled_smpl_features2, gt_smplx_sdf1=gt_smplx_sdf1, gt_smplx_sdf2=gt_smplx_sdf2, mask_box1=mask_box1, mask_box2=mask_box2, normal_box=normal_box, vox1=vox1, vox2=vox2, normal_mask_box1=normal_mask_box1, normal_mask_box2=normal_mask_box2, normal_layered_mask_box1=normal_layered_mask_box1, normal_layered_mask_box2=normal_layered_mask_box2, frontNormal_box1=frontNormal_box1, frontNormal_box2=frontNormal_box2 ) # forward-pass using only the low-resolution PiFU
            netG_output_map1, netG_output_map2 = netG.get_im_feat() # should have shape of [B, 256, H, W]

        net.filter( image_high_tensor1, image_high_tensor2, nmlF1=nmlF_high_tensor1, nmlF2=nmlF_high_tensor2, nmlB1 = nmlB_high_tensor1, nmlB2 = nmlB_high_tensor2, netG_output_map1 = netG_output_map1, netG_output_map2 = netG_output_map2, mask_low_res_tensor1=None, mask_low_res_tensor2=None, mask_high_res_tensor1=mask_high_res_tensor1, mask_high_res_tensor2=mask_high_res_tensor2, unrolled_smpl_features1=None, unrolled_smpl_features2=None, sideview_confidence_box1=sideview_confidence_box1, sideview_confidence_box2=sideview_confidence_box2  ) # forward-pass 
        image_tensor1 = image_high_tensor1
        image_tensor2 = image_high_tensor2

    else:


        if opt.use_mask_for_rendering_low_res:
            mask_low_res_tensor1 = data['mask_low_pifu'].to(device=device)
            mask_low_res_tensor1 = mask_low_res_tensor1.unsqueeze(0)
            mask_low_res_tensor2 = data['mask_low_pifu_side'].to(device=device)
            mask_low_res_tensor2 = mask_low_res_tensor2.unsqueeze(0)
        else:
            mask_low_res_tensor1 = None
            mask_low_res_tensor2 = None


        net.filter( image_low_tensor1, image_low_tensor2, nmlF1=nmlF_low_tensor1, nmlF2=nmlF_low_tensor2, nmlB1 = nmlB_low_tensor1, nmlB2 = nmlB_low_tensor2,  netG_output_map1 = None, netG_output_map2 = None, mask_low_res_tensor1=mask_low_res_tensor1, mask_low_res_tensor2=mask_low_res_tensor2, mask_high_res_tensor1=None, mask_high_res_tensor2=None, unrolled_smpl_features1=unrolled_smpl_features1, unrolled_smpl_features2=unrolled_smpl_features2, gt_smplx_sdf1=gt_smplx_sdf1, gt_smplx_sdf2=gt_smplx_sdf2, mask_box1=mask_box1, mask_box2=mask_box2, normal_box=normal_box, sideview_confidence_box1=sideview_confidence_box1, sideview_confidence_box2=sideview_confidence_box2, vox1=vox1, vox2=vox2, normal_mask_box1=normal_mask_box1, normal_mask_box2=normal_mask_box2, normal_layered_mask_box1=normal_layered_mask_box1, normal_layered_mask_box2=normal_layered_mask_box2, frontNormal_box1=frontNormal_box1, frontNormal_box2=frontNormal_box2, smpl_frontal_depth_values1=smpl_frontal_depth_values1, smpl_frontal_depth_values2=smpl_frontal_depth_values2, layeredNormal_matrix1=layeredNormal_matrix1, layeredNormal_matrix2=layeredNormal_matrix2, layeredNormal_smplx_matrix1=layeredNormal_smplx_matrix1, layeredNormal_smplx_matrix2=layeredNormal_smplx_matrix2  ) # forward-pass 
        image_tensor1 = image_low_tensor1
        image_tensor2 = image_low_tensor2

    

    try:
        save_img_path = save_path[:-4] + '.png'
        save_img_list = []
        for v in range(image_tensor1.shape[0]):
            save_img = (np.transpose(image_tensor1[v].detach().cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, ::-1] * 255.0
            save_img_list.append(save_img)
        save_img = np.concatenate(save_img_list, axis=1)
        cv2.imwrite(save_img_path, save_img)

        if opt.train_single_view_pifu: 
            render_path1 = data['render_path']
            render_path1 = render_path1.split('rendered_image_')[-1]
            render_angle1 = render_path1.replace('.png','')

            render_path2 = data['render_path_side']
            render_path2 = render_path2.split('rendered_image_')[-1]
            render_angle2 = render_path2.replace('.png','')


            save_path1 = save_path.replace(".obj", "_{0}.obj".format(render_angle1) )
            save_path2 = save_path.replace(".obj", "_{0}.obj".format(render_angle2) )

            net.pre_query_single_image(use_frontal=True)
            verts1, faces1, _, _ = reconstruction_singleview(
                net, device, calib_tensor1, resolution, thresh, use_octree=use_octree, num_samples=50000, b_min=b_min1 , b_max=b_max1 , generate_from_low_res = generate_from_low_res)

            verts_tensor1 = torch.from_numpy(verts1.T).unsqueeze(0).to(device=device).float()
            xyz_tensor1 = net.projection(verts_tensor1, calib_tensor1) # verts_tensor should have a range of [-1,1]
            uv1 = xyz_tensor1[:, :2, :]
            color1 = index(image_tensor1, uv1).detach().cpu().numpy()[0].T
            color1 = color1 * 0.5 + 0.5

            save_obj_mesh_with_color(save_path1, verts1, faces1, color1)

            if not do_not_generate_for_sideview:
                # original, for the side view
                net.pre_query_single_image(use_frontal=False)
                verts2, faces2, _, _ = reconstruction_singleview(
                    net, device, calib_tensor2, resolution, thresh, use_octree=use_octree, num_samples=50000, b_min=b_min2 , b_max=b_max2 , generate_from_low_res = generate_from_low_res)

                verts_tensor2 = torch.from_numpy(verts2.T).unsqueeze(0).to(device=device).float()
                xyz_tensor2 = net.projection(verts_tensor2, calib_tensor2) # verts_tensor should have a range of [-1,1]
                uv2 = xyz_tensor2[:, :2, :]
                color2 = index(image_tensor2, uv2).detach().cpu().numpy()[0].T
                color2 = color2 * 0.5 + 0.5

                save_obj_mesh_with_color(save_path2, verts2, faces2, color2)


        else:
            verts, faces, _, _ = reconstruction_multiview(
                net, device, calib_tensor1, calib_tensor2, resolution, thresh, use_octree=use_octree, num_samples=50000, b_min=b_min1 , b_max=b_max1 , generate_from_low_res = generate_from_low_res)

            verts_tensor = torch.from_numpy(verts.T).unsqueeze(0).to(device=device).float()

            xyz_tensor = net.projection(verts_tensor, calib_tensor1) # verts_tensor should have a range of [-1,1]
            uv = xyz_tensor[:, :2, :]
            color = index(image_tensor1, uv).detach().cpu().numpy()[0].T
            color = color * 0.5 + 0.5

            save_obj_mesh_with_color(save_path, verts, faces, color)


    except Exception as e:
        print(e)
        print("Cannot create marching cubes at this time.")






def train(opt):
    global gen_test_counter
    currently_epoch_to_update_low_res_pifu = True
    processes = []
    process_index_to_remove = -1

    if torch.cuda.is_available():
        # set cuda
        #device = 'cuda:0'

        device = 'cuda:{0}'.format(str(opt.gpu_id))

    else:
        device = 'cpu'

    print("using device {}".format(device) )




    train_dataset = TrainDataset(opt, projection='orthogonal', phase = 'train', frontal_only = False, must_generate_sample_pts = not opt.load_sample_pts_from_disk)
    
    projection_mode = train_dataset.projection_mode

    if len(train_dataset) < opt.batch_size:
        batch_size = len(train_dataset)
        print("Change batch_size from {0} to {1}".format(opt.batch_size, batch_size) )
    else:
        batch_size = opt.batch_size
        print("Using batch_size == {0}".format(batch_size) )

    train_data_loader = DataLoader(train_dataset, 
                                   batch_size=opt.batch_size, shuffle=not opt.serial_batches,
                                   num_workers=opt.num_threads, pin_memory=opt.pin_memory)


    print('train loader size: ', len(train_data_loader))


    

    netG = MultiHGPIFuNetwNML(opt, projection_mode, use_High_Res_Component = False)

    if opt.use_High_Res_Component:
        highRes_netG = MultiHGPIFuNetwNML(opt, projection_mode, use_High_Res_Component = True)



    if (not os.path.exists(opt.checkpoints_path) ):
        os.makedirs(opt.checkpoints_path)
    if (not os.path.exists(opt.results_path) ):
        os.makedirs(opt.results_path)
    if (not os.path.exists('%s/%s' % (opt.checkpoints_path, opt.name))  ):
        os.makedirs('%s/%s' % (opt.checkpoints_path, opt.name))
    if (not os.path.exists('%s/%s' % (opt.results_path, opt.name)) ):
        os.makedirs('%s/%s' % (opt.results_path, opt.name))



    if load_model_weights:


        if opt.use_High_Res_Component and opt.activate_HRI_stage:
            pass 
        else: # original

            # load weights for low-res model
            modelG_path = os.path.join( checkpoint_folder_to_load_low_res ,"netG_model_state_dict_epoch{0}.pickle".format(epoch_to_load_from_low_res) )

            print('Resuming from ', modelG_path)

            if device == 'cpu' :
                class CPU_Unpickler(pickle.Unpickler):
                    def find_class(self, module, name):
                        if module == 'torch.storage' and name == '_load_from_bytes':
                            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
                        else:
                            return super().find_class(module, name)

                with open(modelG_path, 'rb') as handle:
                   netG_state_dict = CPU_Unpickler(handle).load()

            else:
                with open(modelG_path, 'rb') as handle:
                   netG_state_dict = pickle.load(handle)

            if do_not_load_low_res_mlp_weights:
                temp_dict = {}
                for k,v in netG_state_dict.items():
                    if 'mlp' not in k:
                        temp_dict[k] = v 
                netG_state_dict = temp_dict

            netG.load_state_dict( netG_state_dict , strict = False )
        
        
        
        # load weights for high-res model
        if opt.use_High_Res_Component and load_model_weights_for_high_res_too:
            
            modelhighResG_path = os.path.join( checkpoint_folder_to_load_high_res, "highRes_netG_model_state_dict_epoch{0}.pickle".format(epoch_to_load_from_high_res) )

            print('Resuming from ', modelhighResG_path)

            if device == 'cpu' :
                class CPU_Unpickler(pickle.Unpickler):
                    def find_class(self, module, name):
                        if module == 'torch.storage' and name == '_load_from_bytes':
                            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
                        else:
                            return super().find_class(module, name)

                with open(modelhighResG_path, 'rb') as handle:
                   highResG_state_dict = CPU_Unpickler(handle).load()

            else:
                with open(modelhighResG_path, 'rb') as handle:
                   highResG_state_dict = pickle.load(handle)


            highRes_netG.load_state_dict( highResG_state_dict , strict = False )
            
        


    opt_log = os.path.join(opt.results_path, opt.name, 'opt.txt')
    with open(opt_log, 'w') as outfile:
        outfile.write(json.dumps(vars(opt), indent=2))



    if opt.use_High_Res_Component and opt.activate_HRI_stage:
        pass 
    else: # original
        netG = netG.to(device=device)
        lr_G = opt.learning_rate_G
        optimizerG = torch.optim.RMSprop(netG.parameters(), lr=lr_G, momentum=0, weight_decay=0)


    

    if (load_model_weights and (not do_not_load_optimizerG_weights) ) and not (opt.use_High_Res_Component and opt.activate_HRI_stage) :
        # load saved weights for optimizerG
        optimizerG_path = os.path.join(checkpoint_folder_to_load_low_res, "optimizerG_epoch{0}.pickle".format(epoch_to_load_from_low_res) )

        if device == 'cpu' :
            with open(optimizerG_path, 'rb') as handle:
                optimizerG_state_dict = CPU_Unpickler(handle).load()

        else:
            with open(optimizerG_path, 'rb') as handle:
                optimizerG_state_dict = pickle.load(handle)


        try:
            optimizerG.load_state_dict( optimizerG_state_dict )
        except Exception as e:
            print(e)
            print("Unable to load optimizerG saved weights!")
        
         
    if opt.use_High_Res_Component:
        highRes_netG = highRes_netG.to(device=device)
        lr_highRes = opt.learning_rate_MR
        optimizer_highRes = torch.optim.RMSprop(highRes_netG.parameters(), lr=lr_highRes, momentum=0, weight_decay=0)
        

        if load_model_weights and load_model_weights_for_high_res_too:
            # load highRes optimizer weights
            optimizer_highRes_path = os.path.join(checkpoint_folder_to_load_high_res, "optimizer_highRes_epoch{0}.pickle".format(epoch_to_load_from_high_res) )
            
            if device == 'cpu' :
                with open(optimizer_highRes_path, 'rb') as handle:
                    optimizer_highRes_state_dict = CPU_Unpickler(handle).load()

            else:
                with open(optimizer_highRes_path, 'rb') as handle:
                    optimizer_highRes_state_dict = pickle.load(handle)



            try:
                optimizer_highRes.load_state_dict( optimizer_highRes_state_dict )
            except Exception as e:
                print(e)
                print("Unable to load optimizer_highRes saved weights!")

            
        
        if opt.update_low_res_pifu and not (opt.use_High_Res_Component and opt.activate_HRI_stage) :
            optimizer_lowResFineTune = torch.optim.RMSprop(netG.parameters(), lr=opt.learning_rate_low_res_finetune, momentum=0, weight_decay=0)

            if load_model_weights and (load_model_weights_for_low_res_finetuning_config != 0):
                # load optimizer_lowResFineTune weights

                if load_model_weights_for_low_res_finetuning_config == 1:
                    optimizer_lowResFineTune_path = os.path.join(checkpoint_folder_to_load_low_res, "optimizerG_epoch{0}.pickle".format(epoch_to_load_from_low_res) )
                elif load_model_weights_for_low_res_finetuning_config == 2:
                    optimizer_lowResFineTune_path = os.path.join(checkpoint_folder_to_load_high_res, "optimizer_lowResFineTune_epoch{0}.pickle".format(epoch_to_load_from_high_res)  )          
                else:
                    raise Exception('Incorrect use of load_model_weights_for_low_res_finetuning_config!')
                

                if device == 'cpu' :
                    with open(optimizer_lowResFineTune_path, 'rb') as handle:
                        optimizer_lowResFineTune_state_dict = CPU_Unpickler(handle).load()

                else:
                    with open(optimizer_lowResFineTune_path, 'rb') as handle:
                        optimizer_lowResFineTune_state_dict = pickle.load(handle)


                optimizer_lowResFineTune.load_state_dict( optimizer_lowResFineTune_state_dict )
                


    for epoch in range(start_epoch, start_epoch + opt.num_epoch):


        print("start of epoch {}".format(epoch) )

        if opt.use_High_Res_Component and opt.activate_HRI_stage:
            pass 
        else:
            netG.train()

        if opt.use_High_Res_Component:
            if opt.update_low_res_pifu:
                if (epoch < opt.epoch_to_start_update_low_res_pifu):
                    currently_epoch_to_update_low_res_pifu = False 
                    print("currently_epoch_to_update_low_res_pifu remains at False for this epoch")
                elif (epoch >= opt.epoch_to_end_update_low_res_pifu):
                    currently_epoch_to_update_low_res_pifu = False
                    print("No longer updating low_res_pifu! In the Finetune Phase") 
                elif (epoch % opt.epoch_interval_to_update_low_res_pifu == 0):
                    currently_epoch_to_update_low_res_pifu = not currently_epoch_to_update_low_res_pifu
                    print("Updating currently_epoch_to_update_low_res_pifu to: ",currently_epoch_to_update_low_res_pifu)
                else:
                    pass

            if opt.update_low_res_pifu and currently_epoch_to_update_low_res_pifu:
                netG.train()
                highRes_netG.eval()
            else:
                if opt.use_High_Res_Component and opt.activate_HRI_stage:
                    pass 
                else:
                    netG.eval()
                highRes_netG.train()


        epoch_error = 0
        train_len = len(train_data_loader)
        for train_idx, train_data in enumerate(train_data_loader):
            print("batch {}".format(train_idx) )

            # retrieve the data
            calib_tensor1 = train_data['calib'].to(device=device) # the calibration matrices for the renders ( is np.matmul(intrinsic, extrinsic)  ). Shape of [Batchsize, 4, 4]
            calib_tensor2 = train_data['calib_side'].to(device=device) # the calibration matrices for the renders ( is np.matmul(intrinsic, extrinsic)  ). Shape of [Batchsize, 4, 4]


            if opt.use_mask_box:
                mask_box1 = train_data['mask_box'].to(device=device)
                mask_box2 = train_data['mask_box_side'].to(device=device)
            else:
                mask_box1 = None
                mask_box2 = None

            if opt.use_frontNormal_box:
                frontNormal_box1 = train_data['frontNormal_box'].to(device=device)
                frontNormal_box2 = train_data['frontNormal_box_side'].to(device=device)
            else:
                frontNormal_box1 = None
                frontNormal_box2 = None


            if opt.train_single_view_pifu and opt.use_normal_box and opt.train_single_view_pifu_activate_firstStage:
                normal_box = train_data['normal_box'].to(device=device) 
            else:
                normal_box = None


            if (opt.train_single_view_pifu and opt.inject_smplx_at_the_start and opt.use_sideview_confidence_box and not opt.train_single_view_pifu_activate_firstStage) or (opt.use_High_Res_Component and opt.activate_HRI_stage and opt.use_sideview_confidence_box_for_HRI):
                sideview_confidence_box1 = train_data['sideview_confidence_box'].to(device=device)
                sideview_confidence_box2 = train_data['sideview_confidence_box_side'].to(device=device)
            else:
                sideview_confidence_box1 = sideview_confidence_box2 = None


            if (opt.use_voxel_ResNet3D and opt.train_single_view_pifu_activate_firstStage and opt.train_single_view_pifu) or opt.use_SeSDF:
                vox1 = train_data['gt_smplx_vox'].to(device=device)
                vox2 = train_data['gt_smplx_vox_side'].to(device=device)

                vox1 = vox1.unsqueeze(1)
                vox2 = vox2.unsqueeze(1)
            else:
                vox1 = vox2 = None

            if opt.use_normal_mask_box_w_3D_CNN:
                normal_mask_box1 = train_data['normal_mask_box'].to(device=device)
                normal_mask_box2 = train_data['normal_mask_box_side'].to(device=device)

            else:
                normal_mask_box1 = normal_mask_box2 = None

            if opt.use_normal_mask_box_in_mlp:
                normal_layered_mask_box1 = train_data['normal_layered_mask_box'].to(device=device)
                normal_layered_mask_box2 = train_data['normal_layered_mask_box_side'].to(device=device)
            else:
                normal_layered_mask_box1 = normal_layered_mask_box2 = None


            if (opt.use_layered_normals and opt.train_single_view_pifu_activate_firstStage and opt.train_single_view_pifu) :
                layeredNormal_matrix1 = train_data['layeredNormal_matrix'].to(device=device)
                layeredNormal_matrix2 = train_data['layeredNormal_matrix_side'].to(device=device)
            else:
                layeredNormal_matrix1 = layeredNormal_matrix2 = None


            if (opt.use_smplx_layered_normals and opt.train_single_view_pifu_activate_firstStage and opt.train_single_view_pifu) :
                layeredNormal_smplx_matrix1 = train_data['layeredNormal_smplx_matrix'].to(device=device)
                layeredNormal_smplx_matrix2 = train_data['layeredNormal_smplx_matrix_side'].to(device=device)
            else:
                layeredNormal_smplx_matrix1 = layeredNormal_smplx_matrix2 = None



            if opt.use_High_Res_Component:
                render_low_pifu_tensor1 = train_data['render_low_pifu'].to(device=device) 
                render_pifu_tensor1 = train_data['original_high_res_render'].to(device=device)  # the renders. Shape of [Batch_size, Channels, Height, Width]
                
                render_low_pifu_tensor2 = train_data['render_low_pifu_side'].to(device=device) 
                render_pifu_tensor2 = train_data['original_high_res_render_side'].to(device=device)  # the renders. Shape of [Batch_size, Channels, Height, Width]
                                
                if opt.use_front_normal:
                    nmlF_low_tensor1 = train_data['nmlF'].to(device=device)
                    nmlF_tensor1 = train_data['nmlF_high_res'].to(device=device)

                    nmlF_low_tensor2 = train_data['nmlF_side'].to(device=device)
                    nmlF_tensor2 = train_data['nmlF_high_res_side'].to(device=device)
                else:
                    nmlF_low_tensor1 = None
                    nmlF_tensor1 = None

                    nmlF_low_tensor2 = None
                    nmlF_tensor2 = None

                if opt.use_back_normal:
                    nmlB_low_tensor1 = train_data['nmlB'].to(device=device)
                    nmlB_tensor1 = train_data['nmlB_high_res'].to(device=device)

                    nmlB_low_tensor2 = train_data['nmlB_side'].to(device=device)
                    nmlB_tensor2 = train_data['nmlB_high_res_side'].to(device=device)
                else:
                    nmlB_low_tensor1 = None
                    nmlB_tensor1 = None

                    nmlB_low_tensor2 = None
                    nmlB_tensor2 = None


            else:

                # low-resolution image that is required by both models
                render_pifu_tensor1 = train_data['render_low_pifu'].to(device=device)  # the renders. Shape of [Batch_size, Channels, Height, Width]
                render_pifu_tensor2 = train_data['render_low_pifu_side'].to(device=device)  # the renders. Shape of [Batch_size, Channels, Height, Width]

                if opt.use_front_normal:
                    nmlF_tensor1 = train_data['nmlF'].to(device=device)
                    nmlF_tensor2 = train_data['nmlF_side'].to(device=device)
                else:
                    nmlF_tensor1 = None
                    nmlF_tensor2 = None

                if opt.use_back_normal:
                    nmlB_tensor1 = train_data['nmlB'].to(device=device)
                    nmlB_tensor2 = train_data['nmlB_side'].to(device=device)
                else:
                    nmlB_tensor1 = None
                    nmlB_tensor2 = None


            gt_pifu_sdf1 = gt_pifu_sdf2 = None # will be set later

            if (opt.inject_smplx_at_the_start) or opt.use_SeSDF or opt.use_3d_trilinear_branch:
                

                if (not opt.replace_inject_smplx_w_inject_pifu) or opt.use_SeSDF or opt.use_3d_trilinear_branch:
                    gt_smplx_sdf1 = train_data['gt_smplx_sdf'].to(device=device)
                    gt_smplx_sdf2 = train_data['gt_smplx_sdf_side'].to(device=device)
                else:
                    gt_smplx_sdf1 = train_data['generated_pifu_sdf'].to(device=device)
                    gt_smplx_sdf2 = train_data['generated_pifu_sdf_side'].to(device=device)

            else:
                gt_smplx_sdf1 = None
                gt_smplx_sdf2 = None




            if opt.use_unrolled_smpl:
                unrolled_smpl_features1 = train_data['unrolled_smpl_features'].to(device=device) 
                unrolled_smpl_features2 = train_data['unrolled_smpl_features_side'].to(device=device) 
            else:
                unrolled_smpl_features1 = None
                unrolled_smpl_features2 = None


            smpl_frontal_depth_values1=None 
            smpl_frontal_depth_values2=None

 
            # low-resolution pifu
            samples_pifu_tensor1 = train_data['samples_low_res_pifu'].to(device=device)  # contain inside and outside points. Shape of [Batch_size, 3, num_of_points]
            labels_pifu_tensor1 = train_data['labels_low_res_pifu'].to(device=device)  # tell us which points in sample_tensor are inside and outside in the surface. Should have shape of [Batch_size ,1, num_of_points]
            if opt.useDOS: 
                samples_pifu_tensor2 = train_data['samples_low_res_pifu_side'].to(device=device)  # contain inside and outside points. Shape of [Batch_size, 3, num_of_points]
                labels_pifu_tensor2 = train_data['labels_low_res_pifu_side'].to(device=device)  # tell us which points in sample_tensor are inside and outside in the surface. Should have shape of [Batch_size ,1, num_of_points]
            else: # original
                samples_pifu_tensor2 = None
                labels_pifu_tensor2 = None



            if opt.use_High_Res_Component:

                if (opt.use_High_Res_Component and opt.activate_HRI_stage):
                    netG_output_map1 = gt_smplx_sdf1
                    netG_output_map2 = gt_smplx_sdf2 
                else: # original
                    netG.filter( render_low_pifu_tensor1, render_low_pifu_tensor2, nmlF1=nmlF_low_tensor1, nmlF2=nmlF_low_tensor2, nmlB1 = nmlB_low_tensor1, nmlB2 = nmlB_low_tensor2, unrolled_smpl_features1=unrolled_smpl_features1, unrolled_smpl_features2=unrolled_smpl_features2, gt_smplx_sdf1=gt_smplx_sdf1, gt_smplx_sdf2=gt_smplx_sdf2, mask_box1=mask_box1, mask_box2=mask_box2, normal_box=normal_box, vox1=vox1, vox2=vox2, normal_mask_box1=normal_mask_box1, normal_mask_box2=normal_mask_box2, normal_layered_mask_box1=normal_layered_mask_box1, normal_layered_mask_box2=normal_layered_mask_box2, frontNormal_box1=frontNormal_box1, frontNormal_box2=frontNormal_box2, layeredNormal_matrix1=layeredNormal_matrix1, layeredNormal_matrix2=layeredNormal_matrix2, layeredNormal_smplx_matrix1=layeredNormal_smplx_matrix1, layeredNormal_smplx_matrix2=layeredNormal_smplx_matrix2 ) # forward-pass using only the low-resolution PiFU
                    netG_output_map1, netG_output_map2 = netG.get_im_feat() # should have shape of [B, 256, H, W]

                if opt.activate_HRI_stage:
                    error_high_pifu, res_high_res_pifu1, res_high_res_pifu2 = highRes_netG.forward(images1=render_pifu_tensor1, images2=render_pifu_tensor2, points=samples_pifu_tensor1, points2=samples_pifu_tensor2, calibs1=calib_tensor1, calibs2=calib_tensor2, labels=labels_pifu_tensor1, labels2=labels_pifu_tensor2,  points_nml=None, labels_nml=None, nmlF1 = nmlF_tensor1, nmlF2 = nmlF_tensor2, nmlB1 = nmlB_tensor1, nmlB2 = nmlB_tensor2, netG_output_map1=netG_output_map1, netG_output_map2=netG_output_map2, sideview_confidence_box1=sideview_confidence_box1, sideview_confidence_box2=sideview_confidence_box2 )
                    res_high_res_pifu = res_high_res_pifu1 # For pointcloud only, set res_high_res_pifu to be results of the frontal image.
                else:
                    error_high_pifu, res_high_res_pifu = highRes_netG.forward(images1=render_pifu_tensor1, images2=render_pifu_tensor2, points=samples_pifu_tensor1, points2=samples_pifu_tensor2, calibs1=calib_tensor1, calibs2=calib_tensor2, labels=labels_pifu_tensor1, labels2=labels_pifu_tensor2, points_nml=None, labels_nml=None, nmlF1 = nmlF_tensor1, nmlF2 = nmlF_tensor2, nmlB1 = nmlB_tensor1, nmlB2 = nmlB_tensor2, netG_output_map1=netG_output_map1, netG_output_map2=netG_output_map2, sideview_confidence_box1=sideview_confidence_box1, sideview_confidence_box2=sideview_confidence_box2 )


                if opt.update_low_res_pifu and currently_epoch_to_update_low_res_pifu and ( not (opt.use_High_Res_Component and opt.activate_HRI_stage) ):
                    optimizer_lowResFineTune.zero_grad()
                    error_high_pifu['Err(occ)'].backward()
                    curr_high_loss = error_high_pifu['Err(occ)'].item()
                    optimizer_lowResFineTune.step() 
                else:
                    optimizer_highRes.zero_grad()
                    error_high_pifu['Err(occ)'].backward()
                    curr_high_loss = error_high_pifu['Err(occ)'].item()
                    optimizer_highRes.step()


                print(
                'Name: {0} | Epoch: {1} | error_high_pifu: {2:.06f} | LR: {3:.06f} '.format(
                    opt.name, epoch, curr_high_loss, lr_highRes)
                )

                epoch_error += curr_high_loss


            else:

                if opt.train_single_view_pifu:
                    error_low_res_pifu, res_low_res_pifu1, res_low_res_pifu2 = netG.forward(images1=render_pifu_tensor1, images2=render_pifu_tensor2, points=samples_pifu_tensor1, points2=samples_pifu_tensor2, calibs1=calib_tensor1, calibs2=calib_tensor2, labels=labels_pifu_tensor1, labels2=labels_pifu_tensor2,  points_nml=None, labels_nml=None, nmlF1 = nmlF_tensor1, nmlF2 = nmlF_tensor2, nmlB1 = nmlB_tensor1, nmlB2 = nmlB_tensor2, unrolled_smpl_features1=unrolled_smpl_features1, unrolled_smpl_features2=unrolled_smpl_features2, gt_smplx_sdf1=gt_smplx_sdf1, gt_smplx_sdf2=gt_smplx_sdf2, gt_pifu_sdf1=gt_pifu_sdf1, gt_pifu_sdf2=gt_pifu_sdf2, mask_box1=mask_box1, mask_box2=mask_box2, normal_box=normal_box, sideview_confidence_box1=sideview_confidence_box1, sideview_confidence_box2=sideview_confidence_box2, vox1=vox1, vox2=vox2, normal_mask_box1=normal_mask_box1, normal_mask_box2=normal_mask_box2, normal_layered_mask_box1=normal_layered_mask_box1, normal_layered_mask_box2=normal_layered_mask_box2, frontNormal_box1=frontNormal_box1, frontNormal_box2=frontNormal_box2, layeredNormal_matrix1=layeredNormal_matrix1, layeredNormal_matrix2=layeredNormal_matrix2, layeredNormal_smplx_matrix1=layeredNormal_smplx_matrix1, layeredNormal_smplx_matrix2=layeredNormal_smplx_matrix2 )
                    res_low_res_pifu = res_low_res_pifu1 # For pointcloud only, set res_low_res_pifu to be results of the frontal image.
                else:
                    error_low_res_pifu, res_low_res_pifu = netG.forward(images1=render_pifu_tensor1, images2=render_pifu_tensor2, points=samples_pifu_tensor1, points2=samples_pifu_tensor2, calibs1=calib_tensor1, calibs2=calib_tensor2, labels=labels_pifu_tensor1, labels2=labels_pifu_tensor2,  points_nml=None, labels_nml=None, nmlF1 = nmlF_tensor1, nmlF2 = nmlF_tensor2, nmlB1 = nmlB_tensor1, nmlB2 = nmlB_tensor2, unrolled_smpl_features1=unrolled_smpl_features1, unrolled_smpl_features2=unrolled_smpl_features2, gt_smplx_sdf1=gt_smplx_sdf1, gt_smplx_sdf2=gt_smplx_sdf2 )
                
                optimizerG.zero_grad()
                error_low_res_pifu['Err(occ)'].backward()
                curr_low_res_loss = error_low_res_pifu['Err(occ)'].item()
                optimizerG.step()

                print(
                'Name: {0} | Epoch: {1} | error_low_res_pifu: {2:.06f} | LR: {3:.06f} '.format(
                    opt.name, epoch, curr_low_res_loss, lr_G)
                )

                epoch_error += curr_low_res_loss




        if opt.use_High_Res_Component and opt.activate_HRI_stage:
            pass 
        else:
            lr_G = adjust_learning_rate(optimizerG, epoch, lr_G, opt.schedule, opt.learning_rate_decay)
        
        if opt.use_High_Res_Component:
            if opt.update_low_res_pifu and currently_epoch_to_update_low_res_pifu and not (opt.use_High_Res_Component and opt.activate_HRI_stage):
                lr_highRes = adjust_learning_rate(optimizer_lowResFineTune, epoch, lr_highRes, opt.schedule, opt.learning_rate_decay)
            else:
                lr_highRes = adjust_learning_rate(optimizer_highRes, epoch, lr_highRes, opt.schedule, opt.learning_rate_decay)


        print("Overall Epoch {0} -  Error for network: {1}".format(epoch, epoch_error/train_len) )


        with torch.no_grad():
            if (  (epoch != 0 and epoch%10==0) or epoch==opt.num_epoch-1   ):

                if save_model_weights and (epoch >= epoch_to_start_saving) and not (opt.use_High_Res_Component and opt.activate_HRI_stage) :
                    # save as pickle:
                    with open( '%s/%s/netG_model_state_dict_epoch%s.pickle' % (opt.checkpoints_path, opt.name, str(epoch) ) , 'wb') as handle:
                        pickle.dump(netG.state_dict(), handle, protocol=pickle.HIGHEST_PROTOCOL)

                    with open( '%s/%s/optimizerG_epoch%s.pickle' % (opt.checkpoints_path, opt.name, str(epoch)) , 'wb') as handle:
                        pickle.dump(optimizerG.state_dict(), handle, protocol=pickle.HIGHEST_PROTOCOL)


                
                if opt.use_High_Res_Component:

                    if generate_point_cloud:
                        r = res_high_res_pifu

                    if save_model_weights and (epoch >= epoch_to_start_saving) :
                        with open( '%s/%s/highRes_netG_model_state_dict_epoch%s.pickle' % (opt.checkpoints_path, opt.name, str(epoch) ) , 'wb') as handle:
                            pickle.dump(highRes_netG.state_dict(), handle, protocol=pickle.HIGHEST_PROTOCOL)

                        with open( '%s/%s/optimizer_highRes_epoch%s.pickle' % (opt.checkpoints_path, opt.name, str(epoch) ) , 'wb') as handle:
                            pickle.dump(optimizer_highRes.state_dict(), handle, protocol=pickle.HIGHEST_PROTOCOL)
                        
                        if opt.update_low_res_pifu and not (opt.use_High_Res_Component and opt.activate_HRI_stage):
                            with open( '%s/%s/optimizer_lowResFineTune_epoch%s.pickle' % (opt.checkpoints_path, opt.name,str(epoch) ) , 'wb') as handle:
                                pickle.dump(optimizer_lowResFineTune.state_dict(), handle, protocol=pickle.HIGHEST_PROTOCOL)

                    highRes_netG.eval()
                else:
                    if generate_point_cloud:
                        r = res_low_res_pifu


                print('generate mesh (train) ...')
                if opt.use_High_Res_Component and opt.activate_HRI_stage:
                    pass 
                else:
                    netG.eval()

                for gen_idx in tqdm(range(1)):

                    index_to_use = gen_test_counter % len(train_dataset_frontal_only)
                    gen_test_counter += 1 
                    train_data = train_dataset_frontal_only.__getitem__(index=index_to_use) 
                    # train_data["img"].shape  has shape of [1, 3, 512, 512]
                    save_path = '%s/%s/train_eval_epoch%d_%s.obj' % (
                        opt.results_path, opt.name, epoch, train_data['name'])


                    generate_from_low_res = True
                    if opt.use_High_Res_Component:
                        gen_mesh(resolution=opt.resolution, net=[netG, highRes_netG] , device = device, data = train_data, save_path = save_path, generate_from_low_res = generate_from_low_res)
                    else:
                        gen_mesh(resolution=opt.resolution, net=netG , device = device, data = train_data, save_path = save_path, generate_from_low_res = generate_from_low_res)

                    






if __name__ == '__main__':

    train(opt)
    


