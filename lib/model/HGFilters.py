'''
MIT License

Copyright (c) 2019 Shunsuke Saito, Zeng Huang, and Ryota Natsume

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''
import numpy as np 
import torch
import torch.nn as nn 
import torch.nn.functional as F 
from ..net_util import conv3x3

class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes, norm='batch'):
        super(ConvBlock, self).__init__()

        # conv3x3 will not change the size (height and width) of the input feature maps 
        self.conv1 = conv3x3(in_planes, int(out_planes / 2))
        self.conv2 = conv3x3(int(out_planes / 2), int(out_planes / 4))
        self.conv3 = conv3x3(int(out_planes / 4), int(out_planes / 4))

        if norm == 'batch':
            self.bn1 = nn.BatchNorm2d(in_planes)
            self.bn2 = nn.BatchNorm2d(int(out_planes / 2))
            self.bn3 = nn.BatchNorm2d(int(out_planes / 4))
            self.bn4 = nn.BatchNorm2d(in_planes)
        elif norm == 'group':
            self.bn1 = nn.GroupNorm(32, in_planes)
            self.bn2 = nn.GroupNorm(32, int(out_planes / 2))
            self.bn3 = nn.GroupNorm(32, int(out_planes / 4))
            self.bn4 = nn.GroupNorm(32, in_planes)
        
        if in_planes != out_planes:
            self.downsample = nn.Sequential(
                self.bn4,
                nn.ReLU(True),
                nn.Conv2d(in_planes, out_planes,
                          kernel_size=1, stride=1, bias=False),
            )
        else:
            self.downsample = None
    
    def forward(self, x):
        residual = x

        out1 = self.conv1(F.relu(self.bn1(x), True))
        out2 = self.conv2(F.relu(self.bn2(out1), True))
        out3 = self.conv3(F.relu(self.bn3(out2), True))

        out3 = torch.cat([out1, out2, out3], 1)

        if self.downsample is not None:
            residual = self.downsample(residual)
        
        out3 += residual

        return out3






class HourGlass(nn.Module):
    def __init__(self, depth, n_features, norm='batch'):
        super(HourGlass, self).__init__()
        self.depth = depth
        self.features = n_features
        self.norm = norm

        self._generate_network(self.depth)

    def _generate_network(self, level):
        self.add_module('b1_' + str(level), ConvBlock(self.features, self.features, norm=self.norm))
        self.add_module('b2_' + str(level), ConvBlock(self.features, self.features, norm=self.norm))

        if level > 1:
            self._generate_network(level - 1)
        else:
            self.add_module('b2_plus_' + str(level), ConvBlock(self.features, self.features, norm=self.norm))

        self.add_module('b3_' + str(level), ConvBlock(self.features, self.features, norm=self.norm))

    def _forward(self, level, inp):
        # upper branch
        up1 = inp 
        up1 = self._modules['b1_' + str(level)](up1)

        # lower branch
        low1 = F.avg_pool2d(inp, 2, stride=2)
        low1 = self._modules['b2_' + str(level)](low1)

        if level > 1:
            low2 = self._forward(level - 1, low1)
        else:
            low2 = low1
            low2 = self._modules['b2_plus_' + str(level)](low2)

        low3 = low2
        low3 = self._modules['b3_' + str(level)](low3)

        up2 = F.interpolate(low3, scale_factor=2, mode='bicubic', align_corners=True)
        # up2 = F.interpolate(low3, scale_factor=2, mode='bilinear')

        return up1 + up2
    
    def forward(self, x):
        return self._forward(self.depth, x)
        










def pixel_unshuffle(input, downscale_factor):
    '''
    input: batchSize * c * k*w * k*h
    kdownscale_factor: k
    batchSize * c * k*w * k*h -> batchSize * k*k*c * w * h
    '''
    c = input.shape[1]

    kernel = torch.zeros(size=[downscale_factor * downscale_factor * c,
                               1, downscale_factor, downscale_factor],
                         device=input.device)
    for y in range(downscale_factor):
        for x in range(downscale_factor):
            kernel[x + y * downscale_factor::downscale_factor*downscale_factor, 0, y, x] = 1
    return F.conv2d(input, kernel, stride=downscale_factor, groups=c)



class HGFilter(nn.Module):
    def __init__(self, stack, depth, in_ch, last_ch, norm='batch', down_type='conv64', use_sigmoid=True, no_first_down_sampling = False, translated_Tanh = None, use_refine_low_res_backbone=False, use_mask_for_low_res_backbone=False, widened_Tanh=None, train_unrolled_gt=False, use_self_attention=False, use_smplx_positional_encoding=False, use_high_resolution_at_start=False, add_input_residual=False, use_modified_Tanh = False, use_injection = False, SDF_Filter_subconfig_except_last=False, SDF_Filter_subconfig_except_penultimate=False):
        super(HGFilter, self).__init__()
        self.n_stack = stack
        self.use_sigmoid = use_sigmoid
        self.depth = depth
        self.last_ch = last_ch # is the no. of channels in the final outputs generated by the HGFilter (i.e. The channel dimension of each element in "outputs"). 
        self.norm = norm
        self.down_type = down_type
        self.no_first_down_sampling = no_first_down_sampling
        self.translated_Tanh = translated_Tanh
        self.widened_Tanh = widened_Tanh
        self.use_refine_low_res_backbone = use_refine_low_res_backbone
        self.train_refine_low_res_backbone = False
        self.use_mask_for_low_res_backbone = use_mask_for_low_res_backbone
        self.train_unrolled_gt = train_unrolled_gt
        self.use_self_attention = use_self_attention
        self.use_smplx_positional_encoding = use_smplx_positional_encoding
        self.do_not_use_vertex_normal = True
        self.use_high_resolution_at_start = use_high_resolution_at_start
        self.add_input_residual = add_input_residual
        self.use_modified_Tanh = use_modified_Tanh
        self.use_injection = use_injection
        self.SDF_Filter_subconfig_except_last = SDF_Filter_subconfig_except_last
        self.SDF_Filter_subconfig_except_penultimate = SDF_Filter_subconfig_except_penultimate

        if self.use_modified_Tanh:
            self.modified_Tanh = modified_Tanh()

        if self.use_high_resolution_at_start:
            in_ch = in_ch * 2 * 2


        if self.use_self_attention:
            in_ch = in_ch + 4 # 4 corresponds to the output channels of self.reduce_conv

            self.embed_dim = 64
            num_heads = 1
            self.number_of_pixels_to_use = 32 * 32 # number of pixels to use
            self.height_of_feat_map = int(np.sqrt(self.number_of_pixels_to_use))
            self.downscale_factor = 16 # scale factor (r) for the pixel unshuffling.

            if not self.use_smplx_positional_encoding:
                self.input_embedding_size = 3 * self.downscale_factor * self.downscale_factor # 3 color channels multiplied by downscale_factor squared after the pixelunshuffling
            else:
                if self.do_not_use_vertex_normal:
                    num_of_extra_feats = 6
                else:
                    num_of_extra_feats = 12
                # the 12 refers to 2 faces x 6 features (3 rest x,y,z and 3 rest normal vector elements)
                self.input_embedding_size = (3+num_of_extra_feats) * self.downscale_factor * self.downscale_factor # 3 color channels multiplied by downscale_factor squared after the pixelunshuffling


            # the three weight matrices must have shape of [input_embedding_size, embed_dim]
            self.w_key = torch.nn.Linear(self.input_embedding_size, self.embed_dim, bias=False)
            self.w_query = torch.nn.Linear(self.input_embedding_size, self.embed_dim, bias=False)
            self.w_value = torch.nn.Linear(self.input_embedding_size, self.embed_dim, bias=False)

            self.multihead_attn = nn.MultiheadAttention(self.embed_dim, num_heads)

            self.reduce_conv = nn.Conv2d(self.embed_dim, 4, kernel_size=3, stride=1, padding=1) # to reduce the channel dimensions during self-attention



        if self.no_first_down_sampling:
            self.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=1, padding=3)

        else:
            self.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3) # will downsample the feature map width and height by half

        last_ch = self.last_ch

        if self.norm == 'batch':
            self.bn1 = nn.BatchNorm2d(64)
        elif self.norm == 'group':
            self.bn1 = nn.GroupNorm(32, 64)

        if self.down_type == 'conv64':
            self.conv2 = ConvBlock(64, 64, self.norm)
            self.down_conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        elif self.down_type == 'conv128':
            self.conv2 = ConvBlock(128, 128, self.norm)
            self.down_conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        elif self.down_type == 'ave_pool' or self.down_type == 'no_down':
            self.conv2 = ConvBlock(64, 128, self.norm)
        
        self.conv3 = ConvBlock(128, 128, self.norm)
        self.conv4 = ConvBlock(128, 256, self.norm)
        


        # start stacking
        for stack in range(self.n_stack):

            self.add_module('m' + str(stack), HourGlass(self.depth, 256, self.norm))

            self.add_module('top_m_' + str(stack), ConvBlock(256, 256, self.norm))
            self.add_module('conv_last' + str(stack),
                            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
            if self.norm == 'batch':
                self.add_module('bn_end' + str(stack), nn.BatchNorm2d(256))
            elif self.norm == 'group':
                self.add_module('bn_end' + str(stack), nn.GroupNorm(32, 256))
            
            self.add_module('l' + str(stack),
                            nn.Conv2d(256, last_ch, 
                            kernel_size=1, stride=1, padding=0))
            if self.train_unrolled_gt:
                self.add_module('unrolled_conv' + str(stack), 
                            nn.Conv2d(last_ch, last_ch, 
                            kernel_size=3, stride=1, padding=1))

            if self.add_input_residual:
                # 3 below represents the RGB channels
                self.add_module(
                    'reduce_residual' + str(stack), nn.Conv2d( 256 + 3*4*4, 256, kernel_size=1, stride=1, padding=0))


            
            if stack < self.n_stack - 1:
                self.add_module(
                    'bl' + str(stack), nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))

                self.add_module(
                    'al' + str(stack), nn.Conv2d(last_ch, 256, kernel_size=1, stride=1, padding=0))

        if self.use_refine_low_res_backbone:
            #self.refine_conv0 = nn.Conv2d(1+256, 0+256, kernel_size=3, stride=1, padding=1)
            self.refine_conv1 = nn.Conv2d(1+256, 256, kernel_size=1, stride=1, padding=0)
            #self.refine_bn0 = nn.InstanceNorm2d(0+256)



    def start_train_refine_low_res_backbone(self):
        self.train_refine_low_res_backbone = True

        for param in self.parameters():
            param.requires_grad = False

        #layers_to_train = [self.refine_conv0, self.refine_conv1, self.refine_bn0]
        layers_to_train = [self.refine_conv1]

        for layer in layers_to_train:
            for param in layer.parameters():
                param.requires_grad = True

    def end_train_refine_low_res_backbone(self):
        self.train_refine_low_res_backbone = False

        for param in self.parameters():
            param.requires_grad = True

        #layers_to_train = [self.refine_conv0, self.refine_conv1, self.refine_bn0]
        layers_to_train = [self.refine_conv1]
        for layer in layers_to_train:
            for param in layer.parameters():
                param.requires_grad = False



    def forward(self, x, gt_smplx_sdf=None):

        # x has shape of (batch_size, 3+?, 512, 512) where 3 == RGB channels; 512 is the width and height
        if self.add_input_residual:
            rgb_unshuffled = x[:,0:3,:,:] # has Shape of [batch, 3, 512,512]
            rgb_unshuffled = pixel_unshuffle(rgb_unshuffled, 4)   # Shape of [batch, 3*4*4 , 128, 128]

        if self.use_high_resolution_at_start:
            # x has Shape of [batch, channels==6, 1024, 1024]
            x = pixel_unshuffle(x, 2)   # Shape of [batch, 6*2*2 , 512, 512]

        if self.use_self_attention:
            # input x must have shape of [number of pixels to use, input embedding size]
            """
            x = [
              [1, 0, 1, 0], # Seq 1
              [0, 2, 0, 2], # Seq 2
              [1, 1, 1, 1]  # Seq 3
             ]
            x = torch.tensor(x, dtype=torch.float32)
            """
            if not self.use_smplx_positional_encoding:
                pixel_unshuffled_x = pixel_unshuffle(x[:,0:3,:,:], self.downscale_factor)   # Shape of [batch, 768, 32, 32]
            else:
                if self.do_not_use_vertex_normal:
                    num_of_extra_feats = 6
                else:
                    num_of_extra_feats = 12
                total_feats = num_of_extra_feats + 3

                # 15 below because 15 = 12 + 3
                pixel_unshuffled_x = pixel_unshuffle(x[:,0:total_feats,:,:], self.downscale_factor)   # Shape of [batch, 2304 or 3840, 32, 32]

                x = torch.cat(  [ x[:,0:3,:,:] , x[:,total_feats:,:,:] ] , dim =1 )

            pixel_unshuffled_x = torch.reshape(pixel_unshuffled_x, (-1, self.input_embedding_size , self.number_of_pixels_to_use ) )  # Shape of [batch, 768, 32*32] 
            pixel_unshuffled_x = torch.transpose(pixel_unshuffled_x, 1, 2) # Shape of [batch, 32*32, 768] where 768 is input embedding size and 32*32 is the number of pixels to use.

            keys = self.w_key(pixel_unshuffled_x)  # Shape of [batch, 32*32, embed_dim] where embed_dim = 64. Note that 32*32 is also the source sequence length
            querys = self.w_query(pixel_unshuffled_x) # Shape of [batch, 32*32, embed_dim] where embed_dim = 64
            values = self.w_value(pixel_unshuffled_x) # Shape of [batch, 32*32, embed_dim] where embed_dim = 64

            # source sequence length (S) == target sequence length (L) == 32*32 for our use-case.
            keys = torch.transpose(keys, 0, 1)     # Shape of (S,N,E) , where S is the source sequence length, N is the batch size, E is the embedding dimension.
            querys = torch.transpose(querys, 0, 1) # Shape of (L,N,E) where L is the target sequence length, N is the batch size, E is the embedding dimension.
            values = torch.transpose(values, 0, 1) # Shape of (S,N,E) , where S is the source sequence length, N is the batch size, E is the embedding dimension.

            # Note that S == L for our use-case
            # attn_output has shape of (L,N,E) where L is the target sequence length, N is the batch size, E is the embedding dimension.
            # attn_output_weights has shape of (N,L,S) where N is the batch size, L is the target sequence length, S is the source sequence length.
            attn_output, attn_output_weights = self.multihead_attn(querys, keys, values)  
            attn_output = torch.transpose(attn_output, 0, 1) 
            attn_output = torch.transpose(attn_output, 1, 2) # now Shape of (N,E,L)  
            attn_output = torch.reshape(attn_output, (-1, self.embed_dim, self.height_of_feat_map, self.height_of_feat_map ) ) # now Shape of (batch, embedded_dim, 32, 32)  

            # upsize the feature map back to 512x512 spatial dimensions
            attn_output = self.reduce_conv(attn_output) # reduce channel dimensions first (reduced from embedded_dim(64) to 4 channels)
            attn_output = F.interpolate(attn_output, scale_factor=self.downscale_factor, mode='bicubic', align_corners=True) # Shape of (batch, 4, 512, 512)  

            x = torch.cat( [x, attn_output], dim=1)





        if self.use_refine_low_res_backbone and self.train_refine_low_res_backbone:
            depth_map = x[:, 9: ,:,:] # "depthmap" here may include human parse map too here if option to use human parse is On.
            depth_map = F.interpolate( depth_map, size=(x.shape[-1] // 4 , x.shape[-1] // 4 ) )
        if self.use_mask_for_low_res_backbone:
            mask = x[:, -1: ,:,:]
            mask = F.interpolate( mask, size=(x.shape[-1] // 4 , x.shape[-1] // 4 ) )
            x = x[:,:-1,:,:]


        x = F.relu(self.bn1(self.conv1(x)), True)

        if self.down_type == 'ave_pool':
            x = F.avg_pool2d(self.conv2(x), 2, stride=2)
        elif self.down_type == ['conv64', 'conv128']:
            x = self.conv2(x)
            x = self.down_conv2(x)
        elif self.down_type == 'no_down':
            x = self.conv2(x)
        else:
            raise NameError('unknown downsampling type')
    
        #normx = x
        normx = 0 

        x = self.conv3(x)
        x = self.conv4(x)

        previous = x

        outputs = []
        for i in range(self.n_stack):

            if self.add_input_residual:
                previous = torch.cat([previous, rgb_unshuffled], dim=1)
                previous = self._modules['reduce_residual' + str(i)](previous)
                    

            hg = self._modules['m' + str(i)](previous)

            ll = hg
            ll = self._modules['top_m_' + str(i)](ll)

            ll = F.relu(self._modules['bn_end' + str(i)]
                       (self._modules['conv_last' + str(i)](ll)), True)

            tmp_out = self._modules['l' + str(i)](ll)
            if self.train_unrolled_gt:
                tmp_out = torch.nn.functional.interpolate(input=tmp_out, scale_factor=2, mode='bilinear', align_corners=True)
                tmp_out = self._modules['unrolled_conv' + str(i)](tmp_out)


            if self.use_sigmoid:
                if self.SDF_Filter_subconfig_except_last: # For this to be True, opt.SDF_Filter must be true and opt.SDF_Filter_config==1 and SDF_Filter_subconfig_except_last set to true
                    if i == self.n_stack - 1:
                        pass 
                    elif self.SDF_Filter_subconfig_except_penultimate and i == self.n_stack - 2:
                        pass 
                    else:
                        tmp_out = nn.Sigmoid()(tmp_out)
                else:
                    tmp_out = nn.Sigmoid()(tmp_out)
            elif self.use_modified_Tanh:
                tmp_out = self.modified_Tanh(tmp_out)
            elif self.translated_Tanh is not None and self.widened_Tanh is None:
                tmp_out = self.translated_Tanh(tmp_out)
            elif self.widened_Tanh is not None:
                tmp_out = self.widened_Tanh(tmp_out)



            if (i == self.n_stack-1) and self.use_refine_low_res_backbone and self.train_refine_low_res_backbone:
                combined_tmp_out = torch.cat( [depth_map, tmp_out], dim=1 )
                #combined_tmp_out = F.leaky_relu(self.refine_bn0(self.refine_conv0(combined_tmp_out)) )
                combined_tmp_out = self.refine_conv1(combined_tmp_out) + tmp_out 

                outputs.append(combined_tmp_out)
            elif (self.use_mask_for_low_res_backbone):
                tmp_out = mask.expand_as(tmp_out) * tmp_out
                outputs.append(tmp_out)
            else:
                outputs.append(tmp_out)


            
            if i < self.n_stack - 1:
                ll = self._modules['bl' + str(i)](ll)
                if self.train_unrolled_gt:
                    tmp_out_ = torch.nn.functional.interpolate(input=tmp_out, scale_factor=0.5, mode='bilinear', align_corners=True)
                    tmp_out_ = self._modules['al' + str(i)](tmp_out_)
                elif self.use_injection:
                    tmp_out_ = tmp_out + gt_smplx_sdf
                    tmp_out_ = self._modules['al' + str(i)](tmp_out)
                else:
                    tmp_out_ = self._modules['al' + str(i)](tmp_out)
                previous = previous + ll + tmp_out_
            
        return outputs, normx



class modified_Tanh(nn.Module):
    def __init__(self):
        super().__init__()
        self.tanh = nn.Tanh()

    def forward(self, x):
        return self.tanh(x) * 1.5 + 0.5


    