# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

def build_model(config):
    model_type = config.MODEL.TYPE
    if model_type == 'swin':
        from .swin_transformer import SwinTransformer
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
                                patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                                reverse_attention_locations=config.MODEL.SWIN.REVERSE_ATTENTION_LOCATIONS)
    elif model_type == 'swin_mlp':
        from .swin_mlp import SwinMLP
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
    elif model_type == 'csam':
        from .csam_transformer import CSAMTransformer
        model = CSAMTransformer(img_size=config.DATA.IMG_SIZE,
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
                                patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                                approach_args=config.MODEL.CSAM
                                )
    elif model_type == 'swin_csamMerge':
        from .swin_transformer_csamPatchMerging import CSAMMergeTransformer
        model = CSAMMergeTransformer(img_size=config.DATA.IMG_SIZE,
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
                                patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                                approach_args=config.MODEL.CSAM
                                )
    elif model_type == 'swin_summary':
        from .swinsummary_transformer import SwinSummaryTransformer
        model = SwinSummaryTransformer(img_size=config.DATA.IMG_SIZE,
                                patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                in_chans=config.MODEL.SWIN.IN_CHANS,
                                num_classes=config.MODEL.NUM_CLASSES,
                                embed_dim=config.MODEL.SWIN.EMBED_DIM,
                                depths=config.MODEL.SWIN.DEPTHS,
                                num_heads=config.MODEL.SWIN.NUM_HEADS,
                                window_size=config.MODEL.SWIN.WINDOW_SIZE,
                                mlp_ratio=float(config.MODEL.SWIN.MLP_RATIO),
                                qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                qk_scale=config.MODEL.SWIN.QK_SCALE,
                                drop_rate=config.MODEL.DROP_RATE,
                                drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                ape=config.MODEL.SWIN.APE,
                                patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                                summary_type=config.MODEL.SWIN.SUMMARY_TYPE, 
                                summary_layers=config.MODEL.SWIN.SUMMARY_LAYERS)
    elif model_type == 'csam_inj_swin':
        from .swin_transformer_with_csam import CSAMInjSwinTransformer
        model = CSAMInjSwinTransformer(img_size=config.DATA.IMG_SIZE,
                                patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                in_chans=config.MODEL.SWIN.IN_CHANS,
                                num_classes=config.MODEL.NUM_CLASSES,
                                embed_dim=config.MODEL.SWIN.EMBED_DIM,
                                depths=config.MODEL.SWIN.DEPTHS,
                                num_heads=config.MODEL.SWIN.NUM_HEADS,
                                window_size=config.MODEL.SWIN.WINDOW_SIZE,
                                mlp_ratio=config.MODEL.SWIN.MLP_RATIO[0] if isinstance(config.MODEL.SWIN.MLP_RATIO, tuple) else config.MODEL.SWIN.MLP_RATIO,
                                qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                qk_scale=config.MODEL.SWIN.QK_SCALE,
                                drop_rate=config.MODEL.DROP_RATE,
                                drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                ape=config.MODEL.SWIN.APE,
                                patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                                approach_args=config.MODEL.CSAM)
    elif model_type  == 'biswin':
        from .reversed_swin_transformer import BiAttnSwinTransformer
        # import pdb; pdb.set_trace()
        model = BiAttnSwinTransformer(
            img_size=config.DATA.IMG_SIZE,
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
            patch_norm=config.MODEL.SWIN.PATCH_NORM,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
            reverse_attention_locations=config.MODEL.SWIN.REVERSE_ATTENTION_LOCATIONS,
            mechanism_instructions={
                'type': config.MODEL.SWIN.ALTERED_ATTENTION.TYPE,
                'reduce_reverse': config.MODEL.SWIN.ALTERED_ATTENTION.REDUCE_REVERSE,
                'reverse_activation': config.MODEL.SWIN.ALTERED_ATTENTION.REVERSE_ACTIVATION,
                'hypernetwork_bias': config.MODEL.SWIN.ALTERED_ATTENTION.HYPERNETWORK_BIAS,
                'project_values': config.MODEL.SWIN.ALTERED_ATTENTION.PROJECT_VALUES,
                'project_input': config.MODEL.SWIN.ALTERED_ATTENTION.PROJECT_INPUT,
                'value_is_input': config.MODEL.SWIN.ALTERED_ATTENTION.VALUE_IS_INPUT,
                'transpose_softmax': config.MODEL.SWIN.ALTERED_ATTENTION.TRANSPOSE_SOFTMAX,
                'activate_hyper_weights': config.MODEL.SWIN.ALTERED_ATTENTION.ACTIVATE_HYPER_WEIGHTS,
                'gen_indiv_hyper_weights': config.MODEL.SWIN.ALTERED_ATTENTION.GEN_INDIV_HYPER_WEIGHTS,
                'activate_input': config.MODEL.SWIN.ALTERED_ATTENTION.ACTIVATE_INPUT,
                'single_weight_matrix': config.MODEL.SWIN.ALTERED_ATTENTION.SINGLE_WEIGHT_MATRIX,
                'weigh_directions': config.MODEL.SWIN.ALTERED_ATTENTION.WEIGH_DIRECTIONS,
                'enforce_orthogonality': config.MODEL.SWIN.ALTERED_ATTENTION.ENFORCE_ORTHONOGALITY,
            }
        )
    elif model_type  == 'invswin':
        from .reversed_swin_transformer import BiAttnSwinTransformer
        model = BiAttnSwinTransformer(
            img_size=config.DATA.IMG_SIZE,
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
            patch_norm=config.MODEL.SWIN.PATCH_NORM,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
            reverse_attention_locations=config.MODEL.SWIN.REVERSE_ATTENTION_LOCATIONS,
            mechanism_instructions={
                'type': config.MODEL.SWIN.ALTERED_ATTENTION.TYPE,
                'reverse_activation': config.MODEL.SWIN.ALTERED_ATTENTION.REVERSE_ACTIVATION,
                'hypernetwork_bias': config.MODEL.SWIN.ALTERED_ATTENTION.HYPERNETWORK_BIAS,
                'selection_lambda_form': config.MODEL.SWIN.ALTERED_ATTENTION.SELECTION_LAMBDA_FORM
            }
        )
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
