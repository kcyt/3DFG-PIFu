
import argparse
import os
import time 

class MultiviewOptions():
    def __init__(self):
        self.initialized = False
        self.parser = None

    def initialize(self, parser):


        # First Stage and Second Stage are run using "train_multiview_pifu.py"
        parser.add_argument('--use_predicted_normal_layered_data', default=True) # Will affect training, validation, and testing, and will affect 'use_layered_normals', 'use_normal_mask_box_w_3D_CNN', and 'use_normal_mask_box_in_mlp'.
        
        # For THuman2.0
        parser.add_argument('--predicted_layered_normal_folder', default="2024-04-07T22-07-37_PredictedResultsForTHumanDataset_for_layered_normal_diffuser_w_normalPriors_wSmplx") # 'use_predicted_normal_layered_data' to be True first.


        parser.add_argument('--restrict_yaw_to_four_angles', default=True) # 'use_predicted_normal_layered_data' to be True first.

        # "train_single_view_pifu" must be True for both First Stage and Second Stage. Set both to False if we training Saito's Multi-view PIFu.
        parser.add_argument('--train_single_view_pifu', default=True)  # Must be True before 'train_single_view_pifu' can be True.
        parser.add_argument('--train_single_view_pifu_activate_firstStage', default=True) # 'train_single_view_pifu' must be True first. If 'train_single_view_pifu_activate_firstStage' is True, then we are using First Stage. If False, we are using Second Stage.

        # 'use_voxel_ResNet3D' should only be used for First Stage (Set 'train_single_view_pifu' and 'train_single_view_pifu_activate_firstStage' to True)
        parser.add_argument('--use_voxel_ResNet3D', default=True)  
        parser.add_argument('--use_3d_trilinear_branch', default=True) #  Whether to add the 3D branch

        parser.add_argument('--use_layered_normals', default=False) #  Will use the gt in the training and will use the predicted (not implemented yet) during testing.
        parser.add_argument('--use_layered_normals_subconfig', default=0) #  if '0' will use the original layered normals; if '1' will use only the front and back normals from the layered normals.
        parser.add_argument('--bug_fix_for_use_layered_normals', default=True)  

        parser.add_argument('--use_smplx_layered_normals', default=False) #  Will use the gt in both training and testing.
        parser.add_argument('--use_smplx_layered_normals_subconfig', default=0) #  if '0' will use the original layered normals; if '1' will use only the front and back normals from the layered normals.
        parser.add_argument('--bug_fix_for_use_smplx_layered_normals', default=True)  

        parser.add_argument('--use_normal_mask_box_w_3D_CNN', default=False) # Use the 4 gt normal maps to form a normal mask box and feed it into a 3D CNN.
        parser.add_argument('--use_normal_mask_box_in_mlp', default=False) # Use the 4 gt normal maps to form a normal mask box, index it, and feed into the MLP.



        # First Stage options. Must be all True if we using First Stage (It is okay they are False if we are doing ablation tests). Must be all set to False if we are using Second Stage. Cannot be used for Saito's Multi-view PIFu 
        parser.add_argument('--use_mask_box', default=False) # Is used for train_multiview_pifu.py.  Mask_box can only be used if 'inject_smplx_at_the_start' is set to True.
        parser.add_argument('--index_mask_box_to_feed_mlp', default=False) # To use, 'use_mask_box' has to be True first.

        parser.add_argument('--use_frontNormal_box', default=False) # Is used for train_multiview_pifu.py. frontNormal_box can only be used if 'inject_smplx_at_the_start' is set to True.
        parser.add_argument('--index_frontNormal_box_to_feed_mlp', default=False) # To use, 'use_frontNormal_box' has to be True first.


        # First Stage and Second Stage options and Can also be used for Saito's Multi-view PIFu (train_multiview_pifu.py):
        parser.add_argument('--inject_smplx_at_the_start', default=False) # Must be True for both First and Second Stage. Must also be true if we are running train_multiview_pifu.py and want to use mask box or side_confidence_box in the filter.
        parser.add_argument('--injected_smplx_size', default=128) # the height and width of the sdf. The depth is always assumed to be 256        
        parser.add_argument('--replace_inject_smplx_w_inject_pifu', default=False) # Must be False for First Stage but True for Second Stage. Only work if 'inject_smplx_at_the_start' is True
        parser.add_argument('--injected_pifu_size', default=256) # only relevant if 'replace_inject_smplx_w_inject_pifu' is True
        parser.add_argument('--do_not_use_smplx_mesh', default=False) # when set as True, will not use the smplx mesh (or the predicted pifu mesh) but will use inputs like mask_box if they are set as True


        # Second Stage options. Cannot be used for Saito's Multi-view PIFu 
        parser.add_argument('--validate_with_sideview_confidence_box', default=False) # Will only work if in Second Stage (i.e. 'train_single_view_pifu' is True and 'train_single_view_pifu_activate_firstStage' is False)
        parser.add_argument('--use_sideview_confidence_box', default=False) # Will only work in Second Stage. Preferred to be True in Second Stage. Will be used in front i.e. as an input to the stacked hourglass. # "train_single_view_pifu" and "inject_smplx_at_the_start" must be True and 'train_single_view_pifu_activate_firstStage' must be False first.


        # To activate HRI stage, set 'train_single_view_pifu', 'train_single_view_pifu_activate_firstStage', and 'activate_HRI_stage' to be True (this is for our models, not Saito's Multi-view PIFu)
        # HRI options ('use_High_Res_Component' and 'activate_HRI_stage') must be set to True first
        # set 'validate_with_sideview_confidence_box' to True if you want to use it here.
        parser.add_argument('--activate_HRI_stage', default=False) # 'inject_smplx_at_the_start' and 'replace_inject_smplx_w_inject_pifu' must be set to True.
        parser.add_argument('--use_sideview_confidence_box_for_HRI', default=False) # this is independent from 'use_sideview_confidence_box'


        parser.add_argument('--env', default="user" )  

        parser.add_argument('--load_sample_pts_from_disk', default=True)  

        parser.add_argument('--useValidationSet', default=True)

        parser.add_argument('--num_epoch', default=60) 
        #parser.add_argument('--num_epoch', default=300) 


        parser.add_argument('--use_unrolled_smpl', default=False) # Must first be true before any of the below sub-options can be true
        parser.add_argument('--output_resolution_smpl_map', default=512)
        parser.add_argument('--use_coordinates_smpl', default=True)  
        parser.add_argument('--use_normals_smpl', default=True) # Must first be true before 'use_normals_smpl_with_rest_pose_normals' can be true
        parser.add_argument('--use_normals_smpl_with_rest_pose_normals', default=True)
        parser.add_argument('--use_blendweights_smpl', default=True)  
        
        # below can apply to either --use_unrolled_smpl or --use_smplx_positional_encoding
        parser.add_argument('--use_gt_smplx', default=True)  # use the gt smplx provided by the owner of THuman2.0 dataset.

        parser.add_argument('--use_High_Res_Component', default=False)  

        parser.add_argument('--update_low_res_pifu', default=False) # Must first be true before any of the below sub-options can be true
        parser.add_argument('--epoch_interval_to_update_low_res_pifu', default=1)
        parser.add_argument('--epoch_to_start_update_low_res_pifu', default=10)
        parser.add_argument('--epoch_to_end_update_low_res_pifu', default=30)


        parser.add_argument('--useDOS', default=False, help='depth oriented sampling') # must be true before any of the useDOS_* options can be true.
        parser.add_argument('--DOS_config_useModifiedDOS', default=True, help='depth oriented sampling') # 'useDOS' must be true first.
        parser.add_argument('--num_of_sets_to_sample', default=4) # default is 4.

        parser.add_argument('--use_mask_for_rendering_high_res',default=False)
        parser.add_argument('--use_mask_for_rendering_low_res',default=False)

        parser.add_argument('--num_threads', default=2, type=int, help='# sthreads for loading data')
        #parser.add_argument('--num_threads', default=4, type=int, help='# sthreads for loading data')

        parser.add_argument('--use_front_normal', default=True)
        parser.add_argument('--use_back_normal', default=True)

        #parser.add_argument('--num_sample_inout', type=int, default=16000, help='# of sampling points')
        parser.add_argument('--num_sample_inout', type=int, default=8000, help='# of sampling points')





        # Less commonly used options:

        parser.add_argument('--use_sideview_confidence_boxes_generated_wo_gtsmplx', default=False) # will only be set by train_multiview_pifu.py
        parser.add_argument('--use_generated_pifu_sdf_generated_wo_gtsmplx', default=False) # will only be set by train_multiview_pifu.py

        parser.add_argument('--use_voxel_ResNet3D_switch_smplx_to_generatedPifuMeshes', default=False)

        parser.add_argument('--use_allViews_dataset', default=False) # will only be set by train_multiview_pifu.py

        parser.add_argument('--gen_samplePts_using_front_side_dataset', default=False)  # will be set by "generate_sample_pts_and_labels.py"
        
        parser.add_argument('--multiview_pifu_config', default=0) # 0 == normal; 1 == configure the mlp. Default = 0

        parser.add_argument('--use_SDFRefinement', default=False) # only for the 'train_multiview_pifu.py' script. Must have 'inject_smplx_at_the_start' and 'replace_inject_smplx_w_inject_pifu' set to be True

        parser.add_argument('--predict_multiple_angles',  default = False)
        parser.add_argument('--config_predict_multiple_angles',  default = 0) # Only relevant if predict_multiple_angles is true and is not using DOS. 0 = only use left and right; 1 = only use left, right, top, and bottom; 2 = use left, right, top, bottom, and behind.
        parser.add_argument('--multiple_angles_use_only_one_MLP',  default = True) # Only relevant if predict_multiple_angles is true.

        parser.add_argument('--use_x_y_offsets_for_MLP', default=False)

        parser.add_argument('--predict_simplified_positional_encoding', default=False)

        parser.add_argument('--predict_colors', default = False)

        parser.add_argument('--use_extremities_module',  default = False)

        parser.add_argument('--predict_frontal_depth',  default = False)

        parser.add_argument('--predict_width',  default = False)

        parser.add_argument('--predict_positional_encoding' ,default = False)

        parser.add_argument('--add_input_residual', default = False) 

        parser.add_argument('--use_high_resolution_at_start', default = False) 

        parser.add_argument('--use_self_attention', default = False) # use self-attention after pixel shuffling.
        
        parser.add_argument('--use_smplx_positional_encoding', default = False)
        

        # below is no longer used
        parser.add_argument('--occlude_rgb_mask_normal', default=False)

        # Corrective Smplx Model training
        parser.add_argument('--use_normal_map_for_smplx_training', default= True)


        parser.add_argument('--sigma_low_resolution_pifu', type=float, default=3.5, help='sigma for sampling')
        parser.add_argument('--sigma_high_resolution_pifu', type=float, default=2.0, help='sigma for sampling') 

        parser.add_argument('--learning_rate_G', type=float, default=1e-3, help='adam learning rate for low res')
        parser.add_argument('--learning_rate_MR', type=float, default=1e-3, help='adam learning rate for high res')




        parser.add_argument('--training_pifuhd', default = False)  # will be auto set to True by trial_train_pifu.py script

        parser.add_argument('--loadSize', type=int, default=1024, help='load size of input image')
        parser.add_argument('--loadSizeGlobal', type=int, default=512, help='load size of input image')

        parser.add_argument('--activate_multi_view', default=True)

        # Experiment related
        timestamp = time.strftime('Date_%d_%b_%y_Time_%H_%M_%S')
        parser.add_argument('--name', type=str, default=timestamp,
                           help='name of the experiment. It decides where to store samples and models')

        # Training related
        parser.add_argument('--tmp_id', type=int, default=0, help='tmp_id')
        parser.add_argument('--gpu_id', type=int, default=0, help='gpu id for cuda')
        
        #parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
        parser.add_argument('--batch_size', type=int, default=2, help='input batch size')

        parser.add_argument('--resolution', type=int, default=256, help='# of grid in mesh reconstruction')
        
        parser.add_argument('--no_first_down_sampling', default=False) # set to True to remove the downsampling

        parser.add_argument('--serial_batches', action='store_true',
                             help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--pin_memory', action='store_true', help='pin_memory')

        parser.add_argument('--ratio_of_way_inside_points',default=0.05) # default is 0.05. Can set to 0.25
        parser.add_argument('--ratio_of_outside_points', default=0.05) # default is 0.05

        parser.add_argument('--learning_rate_decay', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')

        parser.add_argument('--z_size', type=float, default=200.0, help='z normalization factor')


        parser.add_argument('--num_gen_mesh_test', type=int, default=1,
                            help='how many meshes to generate during testing')

        # path
        parser.add_argument('--checkpoints_path', type=str, default='./checkpoints', help='path to save checkpoints')
        parser.add_argument('--results_path', type=str, default='./results', help='path to save results ply')
        
        parser.add_argument('--schedule', default=[99999],
                    help='Decrease learning rate at these epochs.')


        # aug
        group_aug = parser.add_argument_group('aug')
        group_aug.add_argument('--use_augmentation', default=True)
        group_aug.add_argument('--aug_bri', type=float, default=0.2, help='augmentation brightness')
        group_aug.add_argument('--aug_con', type=float, default=0.2, help='augmentation contrast')
        group_aug.add_argument('--aug_sat', type=float, default=0.05, help='augmentation saturation')
        group_aug.add_argument('--aug_hue', type=float, default=0.05, help='augmentation hue')




        parser.add_argument('--num_stack_low_res', type=int, default=4, help='# of hourglass')
        parser.add_argument('--num_stack_high_res', type=int, default=1, help='# of hourglass')

        parser.add_argument('--hg_down', type=str, default='ave_pool', help='ave pool || conv64 || conv128')
        parser.add_argument('--hg_dim_low_res', type=int, default=256, help='256 | 512')
        parser.add_argument('--hg_dim_high_res', type=int, default=16)

        parser.add_argument('--mlp_dim_low_res', nargs='+', default=[257, 1024, 512, 256, 128, 1], type=int,
                             help='# of dimensions of mlp. no need to put the first channel')

        parser.add_argument('--extended_mlp_dim_low_res', nargs='+', default=[257+256+257+256, 1024, 512, 256, 128, 1], type=int,
                             help='# of dimensions of mlp. no need to put the first channel')

        parser.add_argument('--mlp_dim_normal_pred', nargs='+', default=[257, 1024, 512, 256, 128, 3], type=int,
                             help='# of dimensions of mlp. no need to put the first channel')

        parser.add_argument('--mlp_dim_color_pred', nargs='+', default=[257, 1024, 512, 256, 128, 3], type=int,
                             help='# of dimensions of mlp. no need to put the first channel')

        parser.add_argument('--mlp_dim_smplx_pred', nargs='+', default=[257, 1024, 512, 256, 128, 3], type=int,
                             help='# of dimensions of mlp. no need to put the first channel')

        parser.add_argument('--mlp_dim_multiple_angles_pred_low_res', nargs='+', default=[129, 1024, 512, 256, 128, 1], type=int,
                             help='# of dimensions of mlp. no need to put the first channel')
        
        parser.add_argument('--mlp_dim_multiple_angles_pred_high_res', nargs='+', default=[513, 1024, 512, 256, 128, 1], type=int,
                             help='# of dimensions of mlp. no need to put the first channel')

        parser.add_argument('--mlp_res_layers_low_res', nargs='+', default=[2,3,4], type=int,
                             help='leyers that has skip connection. use 0 for no residual pass')

        parser.add_argument('--mlp_res_layers_high_res', nargs='+', default=[1,2], type=int,
                             help='leyers that has skip connection. use 0 for no residual pass')

        parser.add_argument('--merge_layer_low_res', type=int, default=2)

        parser.add_argument('--mlp_dim_high_res', nargs='+', default=[272, 512, 256, 128, 1], type=int,
                             help='# of dimensions of mlp.')

        parser.add_argument('--hg_depth_high_res', type=int, default=2, help='# of stacked layer of hourglass')
        parser.add_argument('--hg_depth_low_res', type=int, default=2, help='# of stacked layer of hourglass')
        parser.add_argument('--mlp_norm', type=str, default='none', help='normalization for volume branch')

        parser.add_argument('--norm', type=str, default='group',
                             help='instance normalization or batch normalization or group normalization')

        # An option for pifuHD only.
        parser.add_argument('--train_full_pifu', default = False)
        parser.add_argument('--no_intermediate_loss', default = False)


        # special tasks
        self.initialized = True
        return parser

    def gather_options(self, args=None):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
            self.parser = parser

        if args is None:
            #return self.parser.parse_args()
            known, unknown = self.parser.parse_known_args()
            return known 
        else:
            #return self.parser.parse_args(args)
            known, unknown = self.parser.parse_known_args(args)
            return known 

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def parse(self, args=None):
        opt = self.gather_options(args)

        opt.home_dir = "/home/XXXX/Documents"

        return opt
