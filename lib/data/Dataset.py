


import os
import random
from math import radians

import numpy as np 
from PIL import Image, ImageOps
import cv2
import torch
import json
import trimesh
import logging

from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn.functional as F


from scipy import sparse 

produce_normal_maps = False 
produce_unrolled_smpl = False
produce_mask_box = False 
produce_gt_smplx = False
produce_gt_smplx_vox = False
produce_generated_pifu_sdf = False
produce_sideview_confidence_boxes = False
product_frontNormal_boxes = False 

NUM_OF_FACES = 6
NUM_OF_INITIAL_FEATURES = 7



def make_rotate(rx, ry, rz):
    # rx is rotation angle about the x-axis
    # ry is rotation angle about the y-axis
    # rz is rotation angle about the z-axis

    sinX = np.sin(rx)
    sinY = np.sin(ry)
    sinZ = np.sin(rz)

    cosX = np.cos(rx)
    cosY = np.cos(ry)
    cosZ = np.cos(rz)

    Rx = np.zeros((3,3))
    Rx[0, 0] = 1.0
    Rx[1, 1] = cosX
    Rx[1, 2] = -sinX
    Rx[2, 1] = sinX
    Rx[2, 2] = cosX

    Ry = np.zeros((3,3))
    Ry[0, 0] = cosY
    Ry[0, 2] = sinY
    Ry[1, 1] = 1.0
    Ry[2, 0] = -sinY
    Ry[2, 2] = cosY

    Rz = np.zeros((3,3))
    Rz[0, 0] = cosZ
    Rz[0, 1] = -sinZ
    Rz[1, 0] = sinZ
    Rz[1, 1] = cosZ
    Rz[2, 2] = 1.0

    R = np.matmul(np.matmul(Rz,Ry),Rx)
    return R


class TestDataset(Dataset):


    def __init__(self, opt):
        self.opt = opt
        self.projection_mode = 'orthogonal'


        self.root = "{0}/render_with_blender/render_results_allViews".format(self.opt.home_dir) 

        self.mask_directory = "{0}/render_with_blender/render_masks_allViews".format(self.opt.home_dir) 


        self.gt_smplx_sdf_directory = "{0}/getSDF_of_SMPLXMeshes/actual_gt_smplx_sdf_results_allViews_w_sdf".format(self.opt.home_dir)  
        self.gt_smplx_vox_directory = "{0}/getSDF_of_SMPLXMeshes/actual_gt_smplx_sdf_results_allViews_voxels".format(self.opt.home_dir)

        if produce_generated_pifu_sdf:
            if self.opt.use_generated_pifu_sdf_generated_wo_gtsmplx:
                self.generated_pifu_sdf_directory = "{0}/generate_PIFu_mesh_SDF/actual_sideview_confidence_boxes_allViews_wo_GTSmplx".format(self.opt.home_dir)
            else:
                self.generated_pifu_sdf_directory = "{0}/generate_PIFu_mesh_SDF/actual_sideview_confidence_boxes_allViews".format(self.opt.home_dir)




        self.subjects = self.get_subjects() # for THuman, this is just "self.training_subject_list"



        # PIL to tensor
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(), #  ToTensor converts input to a shape of (C x H x W) in the range [0.0, 1.0] for each dimension
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # normalise with mean of 0.5 and std_dev of 0.5 for each dimension. Finally range will be [-1,1] for each dimension
        ])


    def __len__(self):
        return len( self.training_subject_list )


    def get_subjects(self):

        return self.training_subject_list 




    def modify_yaw(self, current_yaw, degree_change):
        new_yaw = current_yaw + degree_change

        # converts angles that are more or equal to 360 degree
        if new_yaw>=360:
            new_yaw = new_yaw - 360

        # converts angles that are less 0 degree
        if new_yaw<0:
            new_yaw = new_yaw + 360  

        return new_yaw


    def load_sparse_normal_matrix(self, path, faces, flip_normals, is_smplx=False):
        # faces is either an int ('0' for the first face) or a list (e.g. [0,1] will return a list of two sparse matrices ) 
        # flip_normals is a bool. If True, means the normal matrix is a 180 degree version and the normal vectors need to be flipped.

        current_sparse_matrix = sparse.load_npz(path).todense()
        current_sparse_matrix = np.asarray(current_sparse_matrix) # shape of [OUTPUT_RESOLUTION*OUTPUT_RESOLUTION*6, num_of_features==1] ]
        
        current_sparse_matrix = np.reshape(current_sparse_matrix, [512,512, 6, 3] ) # 6 == no. of faces; 3 = normal vector (normal vector is the only feature); 
        
        if flip_normals:
            # rotate 180 degree about the y-axis
            current_sparse_matrix[:,:,:,0] = -current_sparse_matrix[:,:,:,0] 
            current_sparse_matrix[:,:,:,2] = -current_sparse_matrix[:,:,:,2]
            # flip the width axis of the image
            current_sparse_matrix = np.flip(current_sparse_matrix, axis=1 )

        if isinstance(faces, list):
            temp_matrix = []
            for f in faces: 
                temp_matrix.append( current_sparse_matrix[:,:,f, :] )  # Shape of (512,512, 3)   
            current_sparse_matrix = temp_matrix # will return a list of (512,512,3) elements
        else:
            current_sparse_matrix = current_sparse_matrix[:,:,faces, :]  # (512,512,3) 

        return current_sparse_matrix   


    def load_predicted_sparse_normal_matrix(self, path, faces, flip_normals ):

        current_sparse_matrix = sparse.load_npz(path).todense()
        current_sparse_matrix = np.asarray(current_sparse_matrix) # shape of [6, OUTPUT_RESOLUTION*OUTPUT_RESOLUTION] 
        
        current_sparse_matrix = np.reshape(current_sparse_matrix, [6, 512, 512] ) # 6 == 2 normal maps 
        current_sparse_matrix = np.transpose(current_sparse_matrix, (1,2,0) ) # H,W,C or (512,512,6)

        current_sparse_matrix[current_sparse_matrix<-1.0] = -1.0
        current_sparse_matrix[current_sparse_matrix>1.0] = 1.0

        bool_indices = np.logical_and(current_sparse_matrix>-0.05, current_sparse_matrix<0.05)
        current_sparse_matrix[bool_indices] = 0

        if flip_normals:
            # rotate 180 degree about the y-axis
            current_sparse_matrix[:,:,0] = -current_sparse_matrix[:,:,0] 
            current_sparse_matrix[:,:,2] = -current_sparse_matrix[:,:,2]
            current_sparse_matrix[:,:,3] = -current_sparse_matrix[:,:,3] 
            current_sparse_matrix[:,:,5] = -current_sparse_matrix[:,:,5]
            # flip the width axis of the image
            current_sparse_matrix = np.flip(current_sparse_matrix, axis=1 )


        if isinstance(faces, list):
            temp_matrix = []
            for f in faces:
                temp_matrix.append( current_sparse_matrix[:, :, f*3:(f+1)*3 ] )  # Shape of (512,512, 3)   
        else:
            temp_matrix = current_sparse_matrix[:, :, f*3:(f+1)*3 ]


        return temp_matrix   # each element is of shape (512,512, 3) or H,W,C




    def get_normal_mask_box(self, subject, yaw_front, yaw_rear, yaw_right, yaw_left, downsampled_resolution=128, faces_to_use=[0], path_to_save=None ):

            #if self.evaluation_mode:
            if self.use_predicted_normal_layered_data:

                path_front = "{0}/{4}/subject_{1}_yaw_From_{2:03d}_to_{3:03d}.npz".format(self.opt.home_dir, subject, int(yaw_front), int(yaw_front), self.predicted_layered_normal_folder )
                path_rear = "{0}/{4}/subject_{1}_yaw_From_{2:03d}_to_{3:03d}.npz".format(self.opt.home_dir, subject, int(yaw_front), int(yaw_rear), self.predicted_layered_normal_folder )
                path_right = "{0}/{4}/subject_{1}_yaw_From_{2:03d}_to_{3:03d}.npz".format(self.opt.home_dir, subject, int(yaw_front), int(yaw_right), self.predicted_layered_normal_folder )
                path_left = "{0}/{4}/subject_{1}_yaw_From_{2:03d}_to_{3:03d}.npz".format(self.opt.home_dir, subject, int(yaw_front), int(yaw_left), self.predicted_layered_normal_folder )

                normal_img_front = self.load_predicted_sparse_normal_matrix(path_front, faces=faces_to_use, flip_normals=False) 
                normal_img_rear = self.load_predicted_sparse_normal_matrix(path_rear, faces=faces_to_use, flip_normals=False) 
                normal_img_right = self.load_predicted_sparse_normal_matrix(path_right, faces=faces_to_use, flip_normals=False) 
                normal_img_left = self.load_predicted_sparse_normal_matrix(path_left, faces=faces_to_use, flip_normals=False) 
            else:
                raise Exception("buff cannot use gt normal layered data!")


            if len(faces_to_use) == 1:
                normal_img_front = normal_img_front[0] # [H,W,C]
                normal_img_rear = normal_img_rear[0] # [H,W,C]
                normal_img_right = normal_img_right[0] # [H,W,C]
                normal_img_left = normal_img_left[0] # [H,W,C]
            else:
                normal_img_front = np.concatenate(normal_img_front, axis=2) # [H,W,C]
                normal_img_rear = np.concatenate(normal_img_rear, axis=2)
                normal_img_right = np.concatenate(normal_img_right, axis=2)
                normal_img_left = np.concatenate(normal_img_left, axis=2)

            # start of downsample
            normal_img_front = np.transpose(normal_img_front, (2,0,1) ) # C,H,W
            normal_img_front = torch.tensor(normal_img_front.copy())
            normal_img_front  = F.interpolate(torch.unsqueeze(normal_img_front,0), size=(downsampled_resolution, downsampled_resolution) )
            normal_img_front = normal_img_front[0] # C,H,W
            normal_img_front = normal_img_front.numpy()
            normal_img_front = np.transpose(normal_img_front, (1,2,0) ) # H,W,C

            normal_img_rear = np.transpose(normal_img_rear, (2,0,1) ) # C,H,W
            normal_img_rear = torch.tensor(normal_img_rear.copy())
            normal_img_rear  = F.interpolate(torch.unsqueeze(normal_img_rear,0), size=(downsampled_resolution, downsampled_resolution) )
            normal_img_rear = normal_img_rear[0] # C,H,W
            normal_img_rear = normal_img_rear.numpy()
            normal_img_rear = np.transpose(normal_img_rear, (1,2,0) ) # H,W,C

            normal_img_right = np.transpose(normal_img_right, (2,0,1) ) # C,H,W
            normal_img_right = torch.tensor(normal_img_right.copy())
            normal_img_right  = F.interpolate(torch.unsqueeze(normal_img_right,0), size=(downsampled_resolution, downsampled_resolution) )
            normal_img_right = normal_img_right[0] # C,H,W
            normal_img_right = normal_img_right.numpy()
            normal_img_right = np.transpose(normal_img_right, (1,2,0) ) # H,W,C

            normal_img_left = np.transpose(normal_img_left, (2,0,1) ) # C,H,W
            normal_img_left = torch.tensor(normal_img_left.copy())
            normal_img_left  = F.interpolate(torch.unsqueeze(normal_img_left,0), size=(downsampled_resolution, downsampled_resolution) )
            normal_img_left = normal_img_left[0] # C,H,W
            normal_img_left = normal_img_left.numpy()
            normal_img_left = np.transpose(normal_img_left, (1,2,0) ) # H,W,C
            # end of downsample


            normal_img_front = normal_img_front[:,:,None,:] # [H,W,1,C]
            normal_img_rear = normal_img_rear[:,:,None,:]
            normal_img_right = normal_img_right[:,:,None,:]
            normal_img_left = normal_img_left[:,:,None,:]

            normal_img_front = np.repeat(normal_img_front, downsampled_resolution, axis=2) # [H,W,Z,C] or y,x,z,c
            normal_img_front = np.transpose(normal_img_front, (1, 0, 2, 3) ) # [W,H,Z,C] or x,y,z,c

            normal_img_rear = np.repeat(normal_img_rear, downsampled_resolution, axis=2) # [H,W,Z,C] or y,x,z,c
            normal_img_rear = np.transpose(normal_img_rear, (1, 0, 2, 3) ) # [W,H,Z,C] or x,y,z,c

            normal_img_right = np.repeat(normal_img_right, downsampled_resolution, axis=2) # [H,W,Z,C] or y,x,z,c
            normal_img_right = np.transpose(normal_img_right, (1, 0, 2, 3) ) # [W,H,Z,C] or x,y,z,c in the left's perspective
            normal_img_right = np.transpose(normal_img_right, (2, 1, 0, 3) ) # [W,H,Z,C] or -x,y,z,c in the front's perspective
            normal_img_right = np.flip(normal_img_right, axis=0) # [W,H,Z,C] or x,y,z,c in the front's perspective

            normal_img_left = np.repeat(normal_img_left, downsampled_resolution, axis=2) # [H,W,Z,C] or y,x,z,c
            normal_img_left = np.transpose(normal_img_left, (1, 0, 2, 3) ) # [W,H,Z,C] or x,y,z,c in the left's perspective
            normal_img_left = np.transpose(normal_img_left, (2, 1, 0, 3) ) # [W,H,Z,C] or -x,y,z,c in the front's perspective
            normal_img_left = np.flip(normal_img_left, axis=0) # [W,H,Z,C] or x,y,z,c in the front's perspective


            normal_mask_box = np.concatenate( [normal_img_front, normal_img_rear, normal_img_right, normal_img_left], axis=3 ) # [W,H,Z,4*C] or x,y,z,c in the front's perspective

            # To use in a 3D-CNN or in a MLP for PIFu
            normal_mask_box = np.transpose(normal_mask_box, (3,2,1,0) )  # [4*C,Z,H,W] or [4*C, downsampled_resolution, downsampled_resolution, downsampled_resolution]


            normal_mask_box = torch.tensor(normal_mask_box).float()

            return normal_mask_box





    def get_item(self, index, angle=0):

        side_angle = int(angle) + 90

        # converts angles that are more or equal to 360 degree
        if side_angle>=360:
            side_angle = side_angle - 360

        # converts angles that are less 0 degree
        if side_angle<0:
            side_angle = side_angle + 360       



        # get subject
        subject = self.training_subject_list[index]

        param_path = os.path.join(self.root, "rendered_params_" +  subject + "_{0:03d}".format(int(angle)) + ".npy" ) 
        render_path = os.path.join(self.root, "rendered_image_" +  subject + "_{0:03d}".format(int(angle)) + ".png" ) 
        mask_path = os.path.join(self.mask_directory, "rendered_mask_" +  subject + "_{0:03d}".format(int(angle)) + ".png" ) 

        if produce_normal_maps:
            nmlF_high_res_path =  os.path.join( "{0}/SPIFu/trained_dataset/normal_maps".format(self.opt.home_dir) , "rendered_nmlF_" + subject + "_{0:03d}".format(int(angle)) + ".npz"  )
            nmlB_high_res_path =  os.path.join( "{0}/SPIFu/trained_dataset/normal_maps".format(self.opt.home_dir) , "rendered_nmlB_" + subject + "_{0:03d}".format(int(angle)) + ".npz" )


        if produce_unrolled_smpl:
            unrolled_smpl_features_path = os.path.join(self.unrolled_smpl_features_directory, 'unrolled_smpl_features_sparse_subject_{0}_angle_{1:03d}.npz'.format(subject, int(angle) ) )

        if produce_mask_box:
            mask_box_path = os.path.join(self.mask_boxes_directory, "sparse_mask_box_1_subject_{0}_angle_{1:03d}_from_angle_{2:03d}.npz".format(subject, int(angle), int(side_angle) )  ) 


        if produce_gt_smplx:
            gt_smplx_sdf_path = os.path.join(self.gt_smplx_sdf_directory, "sparse_gt_smplx_sdf_subject_{0}_angle_{0}_{1:03d}.npz".format(subject, int(angle) ) )

            gt_smplx_sdf = sparse.load_npz(gt_smplx_sdf_path).todense()
            gt_smplx_sdf = np.asarray(gt_smplx_sdf) # shape of [128*128, num_of_features] ]
            gt_smplx_sdf = np.reshape(gt_smplx_sdf, [128,128, -1] )  # shape of [128, 128, 256] ] or W,H,Depth (x,y,z) 

            gt_smplx_sdf = torch.tensor(gt_smplx_sdf).float()
            gt_smplx_sdf = gt_smplx_sdf.permute( 2, 1, 0 ) # Depth,H,W i.e. (z,y,x) 
        else:
            gt_smplx_sdf = 0 


        if produce_gt_smplx_vox:
            gt_smplx_vox_path = os.path.join(self.gt_smplx_vox_directory, "sparse_gt_smplx_sdf_subject_{0}_angle_{0}_{1:03d}.npz".format(subject, int(angle) ) )

            gt_smplx_vox = sparse.load_npz(gt_smplx_vox_path).todense()
            gt_smplx_vox = np.asarray(gt_smplx_vox) # shape of [128*128, num_of_features] ]
            gt_smplx_vox = np.reshape(gt_smplx_vox, [128,128, -1] )  # shape of [128, 128, 128] ] or W,H,Depth (x,y,z) 

            gt_smplx_vox = torch.tensor(gt_smplx_vox).float()
            gt_smplx_vox = gt_smplx_vox.permute( 2, 1, 0 ) # Depth,H,W i.e. (z,y,x) 
        else:
            gt_smplx_vox = 0 



        if produce_generated_pifu_sdf:
            generated_pifu_sdf_path = os.path.join(self.generated_pifu_sdf_directory, "sparse_sdf_subject_{0}_angle_{0}_{1:03d}.npz".format(subject, int(angle) ) )  
            generated_pifu_sdf = sparse.load_npz(generated_pifu_sdf_path).todense()
            generated_pifu_sdf = np.asarray(generated_pifu_sdf) # shape of [256*256, num_of_features] ]
            generated_pifu_sdf = np.reshape(generated_pifu_sdf, [256,256, -1] )  # shape of [256, 256, 256] ] or C,H,W i.e. (z,y,x) 
            generated_pifu_sdf = generated_pifu_sdf - 1 # correct the bias that is being added to make the sdf more sparse (see 'option_to_use_save_memory_add_bias' in the actual_gen_PIFu_SDFs.py script)
            generated_pifu_sdf = torch.tensor(generated_pifu_sdf).float()
        else:
            generated_pifu_sdf = 0 



        if produce_sideview_confidence_boxes:
            sideview_confidence_box_path = os.path.join(self.sideview_confidence_box_directory, "sparse_confidence_sideview_box_subject_{0}_angle_{1:03d}.npz".format(subject, int(angle) )  ) 
            sideview_confidence_box = sparse.load_npz(sideview_confidence_box_path).todense()
            sideview_confidence_box = np.asarray(sideview_confidence_box) # shape of [256*256, num_of_features] ]
            sideview_confidence_box = np.reshape(sideview_confidence_box, [256,256, 256] )  # shape of [256, 256, 256] ] W,H,Depth (x,y,z) 
            sideview_confidence_box = torch.tensor(sideview_confidence_box).float()
            sideview_confidence_box = sideview_confidence_box.permute( 2, 1, 0 ) # Depth,H,W i.e. (z,y,x) 
        else:
            sideview_confidence_box = 0


        if product_frontNormal_boxes:
            frontNormal_box_path = os.path.join(self.frontNormal_boxes_directory, "sparse_normal_box_1_subject_{0}_angle_{1:03d}_from_angle_{2:03d}.npz".format(subject, int(angle), int(side_angle) )  ) 
            # to reload back from sparse matrix to numpy array:
            sparse_matrix = sparse.load_npz(frontNormal_box_path).todense()
            frontNormal_box = np.asarray(sparse_matrix)  # shape of [64*64, 256*3]
            frontNormal_box = np.reshape(frontNormal_box, [64, 64, 256, 3] ) # (W=64,H=64,Z=256, 3) or (x,y,z, 3)

            # sidetrack: Need to rotate the normal vectors in normal_box_2 into the angle of space 1.
            angle_difference = int(angle) - int(side_angle)
            rot_mat = make_rotate( 0,  radians(angle_difference) , 0  )
            frontNormal_box = np.reshape(frontNormal_box, [-1, 3] ) # (64*64*256, 3) or (x*y*z, 3)
            frontNormal_box = np.matmul(rot_mat, frontNormal_box.T)  # (3,x*y*z)
            frontNormal_box = frontNormal_box.T  # Shape of (x*y*z, 3)
            frontNormal_box = np.reshape(frontNormal_box, [64, 64, 256, 3] ) # (W=64,H=64,Z=256, 3) or (x,y,z, 3)

            frontNormal_box = torch.tensor(frontNormal_box).float()
            frontNormal_box = frontNormal_box.permute( 3, 2, 1, 0 ) # Channels,Depth,H,W i.e. (3, z,y,x) 

        else:
            frontNormal_box = 0 




        load_size_associated_with_scale_factor = 1024

        # get params
        param = np.load(param_path, allow_pickle=True)  # param is a np.array that looks similar to a dict.  # ortho_ratio = 0.4 , e.g. scale or y_scale = 0.961994278, e.g. center or vmed = [-1.0486  92.56105  1.0101 ]
        center = param.item().get('center') # is camera 3D center position in the 3D World point space (without any rotation being applied).
        R = param.item().get('R')   # R is used to rotate the CAD model according to a given pitch and yaw.
        scale_factor = param.item().get('scale_factor') # is camera 3D center position in the 3D World point space (without any rotation being applied).



        b_range = load_size_associated_with_scale_factor / scale_factor # e.g. 512/scale_factor
        b_center = center
        b_min = b_center - b_range/2
        b_max = b_center + b_range/2


        # the calib matrix is only useful for points of the groundtruth/object mesh. It is not useful for the points under gen_mesh().

        # extrinsic is used to rotate the 3D points according to our specified pitch and yaw
        #translate = -np.matmul(R, center).reshape(3, 1) # because we are translating the 3D points instead of the camera, the translations need to be inverse in this line
        translate = -center.reshape(3, 1)
        extrinsic = np.concatenate([R, translate], axis=1)  # when applied on the 3D pts, the rotation is done first, then the translation
        extrinsic = np.concatenate([extrinsic, np.array([0, 0, 0, 1]).reshape(1, 4)], 0)
        
        temp_extrinsic = np.copy(extrinsic)

        #  scaling actually here, just to invert the y-axis of the image.
        # 'scale_factor' will assume the image is of size 'loadSizeGlobal' i.e. 512 x 512, and this (512 or 1024) does not matter as long as 'uv_intrinsic' is aware of this scaling and thereby ensures that the range is eventually transformed to [-1,1].
        # apparently the scale_intrinsic matrix is to enlarge or magnify the sampled points around or on the surface of the mesh. The range of the sampled surface points is around [-90, 90] after centering, and thus the range is only 180. Yet the range of points allowed by this model (see the uv_intrinsic matrix below) is in [-256, 256]. Thus, we need to scale these sampled surface points up (by multiplying by around 2.4)
        scale_intrinsic = np.identity(4)
        scale_intrinsic[0, 0] = 1.0 * scale_factor #2.4851518#1.0   
        scale_intrinsic[1, 1] = -1.0 * scale_factor #-2.4851518#-1.0
        scale_intrinsic[2, 2] = 1.0 * scale_factor  #2.4851518#1.0

        # Match image pixel space to image uv space  (convert a 512x512 image from range of [-256,255] to range of [-1,1] )
        uv_intrinsic = np.identity(4)
        uv_intrinsic[0, 0] = 1.0 / float(load_size_associated_with_scale_factor // 2) # self.opt.loadSizeGlobal == 512 by default. This value must be 512 unless you change the "scale_factor"
        uv_intrinsic[1, 1] = 1.0 / float(load_size_associated_with_scale_factor // 2) # uv_intrinsic[1, 1] is equal to 1/256
        uv_intrinsic[2, 2] = 1.0 / float(load_size_associated_with_scale_factor // 2) 

        mask = Image.open(mask_path).convert('L') # convert to grayscale (it shd already be grayscale)
        render = Image.open(render_path).convert('RGB')


        # scale_intrinsic inverts the sign of the y-coordinates
        # uv_intrinsic convert the x,y,z coordinates from a 512x512 image into range of [0,2]. * z-coordinates may not be in this range of [0,2] after this transformation )
        intrinsic = np.matmul(uv_intrinsic, scale_intrinsic)
        calib = torch.Tensor(np.matmul(intrinsic, extrinsic)).float() # calib should still work to transform pts into the [-1,1] range even if the input image size actually changes from 512 to 1024.
        extrinsic = torch.Tensor(extrinsic).float()

        mask = transforms.ToTensor()(mask).float()

        render = self.to_tensor(render)  # normalize render from [0,255] to [-1,1]
        render = mask.expand_as(render) * render


        # resize the 1024 x 1024 image to 512 x 512 for the low-resolution pifu
        render_low_pifu = F.interpolate(torch.unsqueeze(render,0), size=(self.opt.loadSizeGlobal,self.opt.loadSizeGlobal) )
        mask_low_pifu = F.interpolate(torch.unsqueeze(mask,0), size=(self.opt.loadSizeGlobal,self.opt.loadSizeGlobal) )
        render_low_pifu = render_low_pifu[0]
        mask_low_pifu = mask_low_pifu[0]

        if produce_normal_maps:
            #nmlF_high_res = np.load(nmlF_high_res_path) # shape of [3, 1024,1024]
            #nmlB_high_res = np.load(nmlB_high_res_path) # shape of [3, 1024,1024]

            # to reload back from sparse matrix to numpy array:
            sparse_matrix = sparse.load_npz(nmlF_high_res_path).todense()
            nmlF_high_res = np.asarray(sparse_matrix) 
            nmlF_high_res = np.reshape(nmlF_high_res, [3, 1024, 1024] )

            sparse_matrix = sparse.load_npz(nmlB_high_res_path).todense()
            nmlB_high_res = np.asarray(sparse_matrix) 
            nmlB_high_res = np.reshape(nmlB_high_res, [3, 1024, 1024] )

            #nmlB_high_res = nmlB_high_res[:,:,::-1].copy()
            nmlF_high_res = torch.Tensor(nmlF_high_res)
            nmlB_high_res = torch.Tensor(nmlB_high_res)
            nmlF_high_res = mask.expand_as(nmlF_high_res) * nmlF_high_res
            nmlB_high_res = mask.expand_as(nmlB_high_res) * nmlB_high_res

            nmlF  = F.interpolate(torch.unsqueeze(nmlF_high_res,0), size=(self.opt.loadSizeGlobal,self.opt.loadSizeGlobal) )
            nmlF = nmlF[0]
            nmlB  = F.interpolate(torch.unsqueeze(nmlB_high_res,0), size=(self.opt.loadSizeGlobal,self.opt.loadSizeGlobal) )
            nmlB = nmlB[0]
        else:
            nmlF_high_res = nmlB_high_res = 0
            nmlF = nmlB = 0 



        smpl_frontal_depth_values = 0 
        features_array = 0 


        if produce_mask_box:

            # to reload back from sparse matrix to numpy array:
            sparse_matrix = sparse.load_npz(mask_box_path).todense()
            mask_box = np.asarray(sparse_matrix) 
            #mask_box = np.reshape(mask_box, [128, 128, 128] ) # (W=128,H=128,Z=128) or (x,y,z) 
            mask_box = np.reshape(mask_box, [256, 256, 256] ) # (W=256,H=256,Z=256) or (x,y,z) 
            mask_box = torch.tensor(mask_box).float()
            mask_box = mask_box.permute( 2, 1, 0 ) # Depth,H,W i.e. (z,y,x) 

            temp_mask = F.interpolate(torch.unsqueeze(mask,0), size=(256,256) )
            temp_mask = temp_mask[0]

            mask_box = temp_mask.expand_as(mask_box) * mask_box    #  (Depth=256,H=256,W=256) i.e. (z,y,x) 

        else:
            mask_box = 0 





        if self.opt.use_normal_mask_box_w_3D_CNN or self.opt.use_normal_mask_box_in_mlp:                       
            rear_angle = self.modify_yaw(angle, 180) 

            right_angle = self.modify_yaw(angle, 270) 

            left_angle = self.modify_yaw(angle, 90) 





        if self.opt.use_normal_mask_box_w_3D_CNN:

            _res_to_use = 128
            if self.use_predicted_normal_layered_data:
                normal_mask_box_path = "{0}/predicted_normal_mask_box_directory/normal_mask_box_subject_{1}_angle_{2:03d}.npz".format(self.opt.home_dir, subject, angle)
            else:
                raise Exception("Buff is not allowed to use GT normal layered data")            

            normal_mask_box = self.get_normal_mask_box(subject=subject, yaw_front=angle, yaw_rear=rear_angle, yaw_right=right_angle, yaw_left=left_angle,
                                                           downsampled_resolution=_res_to_use, faces_to_use=[0], path_to_save=normal_mask_box_path )

        else:
            normal_mask_box = 0




        if self.opt.use_normal_mask_box_in_mlp:
            _res_to_use = 128
            if self.use_predicted_normal_layered_data:
                normal_layered_mask_box_path = "{0}/predicted_normal_layered_mask_box_directory/normal_layered_mask_box_subject_{1}_angle_{2:03d}.npz".format(self.opt.home_dir, subject, angle)
            else:
                raise Exception("Buff is not allowed to use GT normal layered data")
            
            normal_layered_mask_box = self.get_normal_mask_box(subject=subject, yaw_front=angle, yaw_rear=rear_angle, yaw_right=right_angle, yaw_left=left_angle,
                                                                   downsampled_resolution=_res_to_use, faces_to_use=[0,1], path_to_save=normal_layered_mask_box_path )

        else:
            normal_layered_mask_box = 0






        return {
            'name': subject,
            'render_path':render_path,
            'render_low_pifu': render_low_pifu,
            'mask_low_pifu': mask_low_pifu,
            'original_high_res_render':render,
            'mask':mask,
            'calib': calib,
            'nmlF_high_res':nmlF_high_res,
            'nmlB_high_res':nmlB_high_res,
            'nmlF':nmlF,
            'nmlB':nmlB,
            'b_min':b_min,
            'b_max':b_max,
            'unrolled_smpl_features':features_array,
            "smpl_frontal_depth_values":smpl_frontal_depth_values,
            'mask_box':mask_box,
            'gt_smplx_sdf':gt_smplx_sdf,
            'generated_pifu_sdf':generated_pifu_sdf,
            'sideview_confidence_box':sideview_confidence_box,
            'gt_smplx_vox': gt_smplx_vox,
            'frontNormal_box':frontNormal_box,
            'layeredNormal_matrix':layeredNormal_matrix,
            'layeredNormal_smplx_matrix': layeredNormal_smplx_matrix,
            'normal_mask_box':normal_mask_box,
            'normal_layered_mask_box':normal_layered_mask_box

                }



    def __getitem__(self, index):

        final_dict = {}
        final_dict[0] = self.get_item(index, angle=0)
        final_dict[90] = self.get_item(index, angle=90)
        final_dict[180] = self.get_item(index, angle=180)
        final_dict[270] = self.get_item(index, angle=270)

        return final_dict



# for debugging only
if __name__ == "__main__":

    pass





    












