import sys
import torch
from monai.networks.nets import UNet, SwinUNETR
from monai.networks.blocks import UnetOutBlock

def get_model(model_name, in_c=1, n_classes=5, pretrained=False, patch_size=None):
    ## UNET ##
    if model_name == 'small_unet_3d':
        model = UNet(spatial_dims=3, in_channels=in_c, out_channels=n_classes, channels=(16, 32, ), strides=(2, ), num_res_units=1, )
    elif model_name == 'tiny_swin_unetr':
        feat_sz = 12
        if pretrained:  # weights from training on btcv, its ct but meh
            model = SwinUNETR(img_size=patch_size, in_channels=in_c, out_channels=14, feature_size=feat_sz, use_checkpoint=False)
            state_dict = torch.load('pretrained_weights/swin_unetr_tiny_btcv.pt')["state_dict"]
            model.load_state_dict(state_dict)

            # grab pretrained weights of input layer(s), which have one channel
            input_weight_name1 = 'swinViT.patch_embed.proj.weight'
            input_weight_name2 = 'encoder1.layer.conv1.conv.weight'
            input_weight_name3 = 'encoder1.layer.conv3.conv.weight'

            input_weight1 = state_dict[input_weight_name1]
            input_weight2 = state_dict[input_weight_name2]
            input_weight3 = state_dict[input_weight_name3]

            # stack as many times as wanted input channels, manipulate the pretrained state_dict
            state_dict[input_weight_name1] = torch.cat(in_c * [input_weight1], dim=1)
            state_dict[input_weight_name2] = torch.cat(in_c * [input_weight2], dim=1)
            state_dict[input_weight_name3] = torch.cat(in_c * [input_weight3], dim=1)
            # now we can load our model

            print('successfully loaded pretrained weights for swinunetr_tiny')
            model.out = UnetOutBlock(spatial_dims=3, in_channels=feat_sz, out_channels=n_classes)
        else:
            model = SwinUNETR(img_size=patch_size, in_channels=in_c, out_channels=n_classes, feature_size=feat_sz)

    else:
        sys.exit('not a valid model_name, check utils.get_model.py')

    setattr(model, 'n_classes', n_classes)
    setattr(model, 'patch_size', patch_size)

    return model