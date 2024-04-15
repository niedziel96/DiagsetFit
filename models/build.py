# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from .swin_transformer import SwinTransformer
from .swin_transformer_v2 import SwinTransformerV2
from .swin_transformer_moe import SwinTransformerMoE
from .swin_mlp import SwinMLP
from .simmim import build_simmim
from .dinov2 import build_dinov2
import torchvision
import timm
from .fVIT import _vision_transformer
import torch 
from .ensemble import EnsembleModel

def build_model(config, is_pretrain=False):
    model_type = config.MODEL.TYPE

    # accelerate layernorm
    if config.FUSED_LAYERNORM:
        try:
            import apex as amp
            layernorm = amp.normalization.FusedLayerNorm
        except:
            layernorm = None
            print("To use FusedLayerNorm, please install apex.")
    else:
        import torch.nn as nn
        layernorm = nn.LayerNorm

    if is_pretrain:
        model = build_simmim(config)
        return model

    if model_type == 'swin':
        print(config.DATA.IMG_SIZE)
        model = SwinTransformer(img_size=config.DATA.IMG_SIZE,
                                patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                in_chans=config.MODEL.SWIN.IN_CHANS,
                                num_classes=config.MODEL.NUM_CLASSES,
                                embed_dim=config.MODEL.SWIN.EMBED_DIM,
                                depths=config.MODEL.SWIN.DEPTHS,
                                num_heads=config.MODEL.SWIN.NUM_HEADS,
                                window_size=config.MODEL.SWIN.WINDOW_SIZE,
                                mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                                qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                qk_scale=config.MODEL.SWIN.QK_SCALE,
                                drop_rate=config.MODEL.DROP_RATE,
                                drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                ape=config.MODEL.SWIN.APE,
                                norm_layer=layernorm,
                                patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                                fused_window_process=config.FUSED_WINDOW_PROCESS)
    elif model_type == 'swinv2':
        model = SwinTransformerV2(img_size=config.DATA.IMG_SIZE,
                                  patch_size=config.MODEL.SWINV2.PATCH_SIZE,
                                  in_chans=config.MODEL.SWINV2.IN_CHANS,
                                  num_classes=config.MODEL.NUM_CLASSES,
                                  embed_dim=config.MODEL.SWINV2.EMBED_DIM,
                                  depths=config.MODEL.SWINV2.DEPTHS,
                                  num_heads=config.MODEL.SWINV2.NUM_HEADS,
                                  window_size=config.MODEL.SWINV2.WINDOW_SIZE,
                                  mlp_ratio=config.MODEL.SWINV2.MLP_RATIO,
                                  qkv_bias=config.MODEL.SWINV2.QKV_BIAS,
                                  drop_rate=config.MODEL.DROP_RATE,
                                  drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                  ape=config.MODEL.SWINV2.APE,
                                  patch_norm=config.MODEL.SWINV2.PATCH_NORM,
                                  use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                                  pretrained_window_sizes=config.MODEL.SWINV2.PRETRAINED_WINDOW_SIZES)
    elif model_type == 'swin_moe':
        model = SwinTransformerMoE(img_size=config.DATA.IMG_SIZE,
                                   patch_size=config.MODEL.SWIN_MOE.PATCH_SIZE,
                                   in_chans=config.MODEL.SWIN_MOE.IN_CHANS,
                                   num_classes=config.MODEL.NUM_CLASSES,
                                   embed_dim=config.MODEL.SWIN_MOE.EMBED_DIM,
                                   depths=config.MODEL.SWIN_MOE.DEPTHS,
                                   num_heads=config.MODEL.SWIN_MOE.NUM_HEADS,
                                   window_size=config.MODEL.SWIN_MOE.WINDOW_SIZE,
                                   mlp_ratio=config.MODEL.SWIN_MOE.MLP_RATIO,
                                   qkv_bias=config.MODEL.SWIN_MOE.QKV_BIAS,
                                   qk_scale=config.MODEL.SWIN_MOE.QK_SCALE,
                                   drop_rate=config.MODEL.DROP_RATE,
                                   drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                   ape=config.MODEL.SWIN_MOE.APE,
                                   patch_norm=config.MODEL.SWIN_MOE.PATCH_NORM,
                                   mlp_fc2_bias=config.MODEL.SWIN_MOE.MLP_FC2_BIAS,
                                   init_std=config.MODEL.SWIN_MOE.INIT_STD,
                                   use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                                   pretrained_window_sizes=config.MODEL.SWIN_MOE.PRETRAINED_WINDOW_SIZES,
                                   moe_blocks=config.MODEL.SWIN_MOE.MOE_BLOCKS,
                                   num_local_experts=config.MODEL.SWIN_MOE.NUM_LOCAL_EXPERTS,
                                   top_value=config.MODEL.SWIN_MOE.TOP_VALUE,
                                   capacity_factor=config.MODEL.SWIN_MOE.CAPACITY_FACTOR,
                                   cosine_router=config.MODEL.SWIN_MOE.COSINE_ROUTER,
                                   normalize_gate=config.MODEL.SWIN_MOE.NORMALIZE_GATE,
                                   use_bpr=config.MODEL.SWIN_MOE.USE_BPR,
                                   is_gshard_loss=config.MODEL.SWIN_MOE.IS_GSHARD_LOSS,
                                   gate_noise=config.MODEL.SWIN_MOE.GATE_NOISE,
                                   cosine_router_dim=config.MODEL.SWIN_MOE.COSINE_ROUTER_DIM,
                                   cosine_router_init_t=config.MODEL.SWIN_MOE.COSINE_ROUTER_INIT_T,
                                   moe_drop=config.MODEL.SWIN_MOE.MOE_DROP,
                                   aux_loss_weight=config.MODEL.SWIN_MOE.AUX_LOSS_WEIGHT)
    elif model_type == 'swin_mlp':
        model = SwinMLP(img_size=config.DATA.IMG_SIZE,
                        patch_size=config.MODEL.SWIN_MLP.PATCH_SIZE,
                        in_chans=config.MODEL.SWIN_MLP.IN_CHANS,
                        num_classes=config.MODEL.NUM_CLASSES,
                        embed_dim=config.MODEL.SWIN_MLP.EMBED_DIM,
                        depths=config.MODEL.SWIN_MLP.DEPTHS,
                        num_heads=config.MODEL.SWIN_MLP.NUM_HEADS,
                        window_size=config.MODEL.SWIN_MLP.WINDOW_SIZE,
                        mlp_ratio=config.MODEL.SWIN_MLP.MLP_RATIO,
                        drop_rate=config.MODEL.DROP_RATE,
                        drop_path_rate=config.MODEL.DROP_PATH_RATE,
                        ape=config.MODEL.SWIN_MLP.APE,
                        patch_norm=config.MODEL.SWIN_MLP.PATCH_NORM,
                        use_checkpoint=config.TRAIN.USE_CHECKPOINT)
    elif model_type == 'vit_b_32':
        model = torchvision.models.vit_b_32(weights='IMAGENET1K_V1')
        num_ftrs = model.heads.head.in_features
        model.heads.head = nn.Linear(num_ftrs, config.MODEL.VIT.NUM_CLASSES)
    elif model_type == 'vit_b_16':
        model = torchvision.models.vit_b_16(weights='IMAGENET1K_V1')
        num_ftrs = model.heads.head.in_features
        model.heads.head = nn.Linear(num_ftrs, config.MODEL.VIT.NUM_CLASSES)
    elif model_type == 'vit_l_16':
        model = torchvision.models.vit_l_16(weights='IMAGENET1K_V1')
        num_ftrs = model.heads.head.in_features
        model.heads.head = nn.Linear(num_ftrs, config.MODEL.VIT.NUM_CLASSES)
    elif model_type == 'tiny_vit':
        model = timm.create_model('tiny_vit_21m_224.dist_in22k_ft_in1k', pretrained=True)
        num_ftrs = model.head.fc.in_features
        model.head.fc = nn.Linear(num_ftrs, config.MODEL.VIT.NUM_CLASSES)
    elif model_type == 'tiny_vit_11m':
        model = timm.create_model('tiny_vit_11m_224.dist_in22k_ft_in1k', pretrained=True)
        num_ftrs = model.head.fc.in_features
        model.head.fc = nn.Linear(num_ftrs, config.MODEL.VIT.NUM_CLASSES)
    elif model_type == 'tiny_vit_5m':
        model = timm.create_model('tiny_vit_5m_224.dist_in22k_ft_in1k', pretrained=True)
        num_ftrs = model.head.fc.in_features
        model.head.fc = nn.Linear(num_ftrs, config.MODEL.VIT.NUM_CLASSES)
    elif model_type == 'vit_l_32':
        model = torchvision.models.vit_l_32(weights='IMAGENET1K_V1')
        num_ftrs = model.heads.head.in_features
        model.heads.head = nn.Linear(num_ftrs, config.MODEL.VIT.NUM_CLASSES)
    elif model_type == 'dinov2_small':
        model = build_dinov2(model_type, True, config.MODEL.VIT.NUM_CLASSES)
    elif model_type == 'dinov2_base':
        model = build_dinov2(model_type, True, config.MODEL.VIT.NUM_CLASSES)
    elif model_type == 'dinov2_large':
        model = build_dinov2(model_type, True, config.MODEL.VIT.NUM_CLASSES)
    elif model_type == 'swinv2_tiny':
        model = torchvision.models.swin_v2_t(weights='IMAGENET1K_V1')
        num_ftrs = model.head.in_features
        model.head = nn.Linear(num_ftrs, config.MODEL.VIT.NUM_CLASSES)
    elif model_type == 'swinv2_base':
        model = torchvision.models.swin_v2_b(weights='IMAGENET1K_V1')
        num_ftrs = model.head.in_features
        model.head = nn.Linear(num_ftrs, config.MODEL.VIT.NUM_CLASSES)
    elif model_type == 'swinv2_small':
        model = torchvision.models.swin_v2_s(weights='IMAGENET1K_V1')
        num_ftrs = model.head.in_features
        model.head = nn.Linear(num_ftrs, config.MODEL.VIT.NUM_CLASSES)
    elif model_type == 'vgg_19':
        model = torchvision.models.vgg19(weights='IMAGENET1K_V1')
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, config.MODEL.VIT.NUM_CLASSES)
    elif model_type == 'ensemble_A':

        # create base for vit_tiny 5m 
        model_vit_tiny_5m = timm.create_model('tiny_vit_5m_224.dist_in22k_ft_in1k', pretrained=True)
        num_ftrs_tiny = model_vit_tiny_5m.head.fc.in_features
        model_vit_tiny_5m.head.fc = nn.Linear(num_ftrs_tiny, config.MODEL.VIT.NUM_CLASSES)

        # now load best weights for vit tiny  
        chckpts_tiny = torch.load(config.MODEL.ENSEMBLE.CH_1, map_location='cpu')
        model_vit_tiny_5m.load_state_dict(chckpts_tiny['model'], strict=False) # should be alright 

        # create base for vit_b_16
        # model_vit_b_16 = torchvision.models.vit_b_16(weights='IMAGENET1K_V1')
        # num_ftrs_b16 = model_vit_b_16.heads.head.in_features
        # model_vit_b_16.heads.head = nn.Linear(num_ftrs_b16, config.MODEL.VIT.NUM_CLASSES)

        # now load best weights for vit tiny  
        # chckpts_b_16 = torch.load(config.MODEL.ENSEMBLE.CH_2, map_location='cpu')
        # model_vit_b_16.load_state_dict(chckpts_b_16['model'], strict=False) # should be alright 

        # # create last model - dino 
        # model_dinov2_b = build_dinov2('dinov2_base', True, config.MODEL.VIT.NUM_CLASSES)

        # # load wights for dino: 
        # chckpts_dino = torch.load(config.MODEL.ENSEMBLE.CH_3, map_location='cpu')
        # model_dinov2_b.load_state_dict(chckpts_dino['model'], strict=False) # should be alright 

        model_vgg = torchvision.models.vgg19(weights='IMAGENET1K_V1')
        num_ftrs = model_vgg.classifier[6].in_features
        model_vgg.classifier[6] = nn.Linear(num_ftrs, config.MODEL.VIT.NUM_CLASSES)
        chckpts_vgg = torch.load(config.MODEL.ENSEMBLE.CH_3, map_location='cpu')
        model_vgg.load_state_dict(chckpts_vgg['model'], strict=False) # should be alright 

        model = EnsembleModel(model_vit_tiny_5m, model_vgg)

        for param in model.parameters():
            param.requires_grad = False

        for param in model.classifier.parameters():
            param.requires_grad = True    

    elif model_type == 'ensemble_B':

        # create base for vit_tiny 5m 
        model_vit_tiny_5m = timm.create_model('tiny_vit_5m_224.dist_in22k_ft_in1k', pretrained=True)
        num_ftrs_tiny = model_vit_tiny_5m.head.fc.in_features
        model_vit_tiny_5m.head.fc = nn.Linear(num_ftrs_tiny, config.MODEL.VIT.NUM_CLASSES)

        # now load best weights for vit tiny  
        chckpts_tiny = torch.load(config.MODEL.ENSEMBLE.CH_1, map_location='cpu')
        model_vit_tiny_5m.load_state_dict(chckpts_tiny['model'], strict=False) # should be alright 

        # create base for vit_b_16
        model_vit_b_16 = torchvision.models.vit_b_16(weights='IMAGENET1K_V1')
        num_ftrs_b16 = model_vit_b_16.heads.head.in_features
        model_vit_b_16.heads.head = nn.Linear(num_ftrs_b16, config.MODEL.VIT.NUM_CLASSES)

        # now load best weights for vit tiny  
        chckpts_b_16 = torch.load(config.MODEL.ENSEMBLE.CH_2, map_location='cpu')
        model_vit_b_16.load_state_dict(chckpts_b_16['model'], strict=False) # should be alright 



        model = EnsembleModel(model_vit_tiny_5m, model_vit_b_16)

        for param in model.parameters():
            param.requires_grad = False

        for param in model.classifier.parameters():
            param.requires_grad = True   

    elif model_type == 'experimental_vit_b_16_v1':
        # version from img -> feature -> result | V1 

        model = _vision_transformer(
        patch_size=16,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        pretrained=False,
        progress=True,
        arch='vit_b_16')

        # model = torchvision.models.vit_b_16(weights='IMAGENET1K_V1')
        num_ftrs = model.heads.head.in_features
        model.heads.head = nn.Linear(num_ftrs, config.MODEL.VIT.NUM_CLASSES)
        # load weights here: 
        chckpts = torch.load(config.MODEL.checkpoints, map_location='cpu')
        model.load_state_dict(chckpts['model'], strict=False)

        # edit model 
        model.seq_length = 784# feature dependent - to be updated 
        hidden_dim = model.encoder.pos_embedding.shape[2]
        model.encoder.pos_embedding =  nn.Parameter(torch.empty(1, 785, hidden_dim).normal_(std=0.02)) 
        model.conv_proj = nn.Linear(768,768) # should work??? <- it's hardcoded ofc :))))

    elif model_type == 'experimental_vit_b_16_v2':
        # version from img -> feature -> result | V1 

        model = _vision_transformer(
        patch_size=16,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        pretrained=False,
        progress=True,
        arch='vit_b_16')

        if 'imagenet' in config.MODEL.checkpoints:
            chckpts = torch.load(config.MODEL.checkpoints, map_location='cpu')
            model.load_state_dict(chckpts, strict=False)

        # model = torchvision.models.vit_b_16(weights='IMAGENET1K_V1')
        num_ftrs = model.heads.head.in_features
        model.heads.head = nn.Linear(num_ftrs, config.MODEL.VIT.NUM_CLASSES)
        
        if 'imagenet' not in config.MODEL.checkpoints: # load weights here: 
            chckpts = torch.load(config.MODEL.checkpoints, map_location='cpu')
            model.load_state_dict(chckpts['model'], strict=False)

        # edit model 
        model.seq_length = 784# feature dependent - to be updated 
        hidden_dim = model.encoder.pos_embedding.shape[2]
        model.encoder.pos_embedding =  nn.Parameter(torch.empty(1, 785, hidden_dim).normal_(std=0.02)) 
        model.conv_proj = nn.Sequential(*[nn.Linear(768, 768), nn.GELU(), nn.Dropout(p=0.)])

    elif model_type == 'fVIT_b_16':
        model = _vision_transformer(
        patch_size=16,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        pretrained=False,
        progress=True,
        arch='vit_b_16'
    )
        num_ftrs = model.heads.head.in_features
        model.heads.head = nn.Linear(num_ftrs, config.MODEL.VIT.NUM_CLASSES)
        
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
