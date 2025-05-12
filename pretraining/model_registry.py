from ecog_foundation_model.config import ViTConfig

# Registry of functions for creating model configs.
model_registry = {}


def register_model(name=None):
    """
    Decorator to register a model constructor function used for decoding models.

    The decorated function must follow the signature:
        model_fn() -> ViTConfig

    Returns:
        function: A decorator that registers the model constructor in
                  model_constructor_registry.
    """

    def decorator(fn):
        model_name = name or fn.__name__
        model_registry[model_name] = fn
        return fn

    return decorator


@register_model()
def patch_dim_2_small():
    return ViTConfig(
        dim=128,
        decoder_embed_dim=64,
        mlp_ratio=4.0,
        depth=6,
        decoder_depth=4,
        num_heads=4,
        decoder_num_heads=4,
        patch_size=2,
        frame_patch_size=16,
        use_cls_token=False,
        sep_pos_embed=True,
        trunc_init=False,
        no_qkv_bias=False,
        proj_drop=0.1,
        drop_path=0.05,
    )


@register_model()
def patch_dim_2_medium():
    return ViTConfig(
        dim=384,
        decoder_embed_dim=192,
        mlp_ratio=4.0,
        depth=10,
        decoder_depth=6,
        num_heads=6,
        decoder_num_heads=4,
        patch_size=2,
        frame_patch_size=8,
        use_cls_token=False,
        sep_pos_embed=True,
        trunc_init=False,
        no_qkv_bias=False,
        proj_drop=0.1,
        drop_path=0.1,
    )


@register_model()
def patch_dim_2_large():
    return ViTConfig(
        dim=512,
        decoder_embed_dim=256,
        mlp_ratio=4.0,
        depth=12,
        decoder_depth=8,
        num_heads=8,
        decoder_num_heads=4,
        patch_size=2,
        frame_patch_size=4,
        use_cls_token=False,
        sep_pos_embed=True,
        trunc_init=False,
        no_qkv_bias=False,
        proj_drop=0.2,
        drop_path=0.1,
    )


@register_model()
def patch_dim_1_small():
    return ViTConfig(
        dim=128,
        decoder_embed_dim=64,
        mlp_ratio=4.0,
        depth=6,
        decoder_depth=4,
        num_heads=4,
        decoder_num_heads=4,
        patch_size=1,
        frame_patch_size=16,
        use_cls_token=False,
        sep_pos_embed=True,
        trunc_init=False,
        no_qkv_bias=False,
        proj_drop=0.1,
        drop_path=0.05,
    )


@register_model()
def patch_dim_1_medium():
    return ViTConfig(
        dim=256,
        decoder_embed_dim=128,
        mlp_ratio=4.0,
        depth=8,
        decoder_depth=4,
        num_heads=4,
        decoder_num_heads=4,
        patch_size=1,
        frame_patch_size=8,
        use_cls_token=False,
        sep_pos_embed=True,
        trunc_init=False,
        no_qkv_bias=False,
        proj_drop=0.1,
        drop_path=0.1,
    )


@register_model()
def patch_dim_1_large():
    return ViTConfig(
        dim=384,
        decoder_embed_dim=192,
        mlp_ratio=4.0,
        depth=10,
        decoder_depth=6,
        num_heads=6,
        decoder_num_heads=4,
        patch_size=1,
        frame_patch_size=8,
        use_cls_token=False,
        sep_pos_embed=True,
        trunc_init=False,
        no_qkv_bias=False,
        proj_drop=0.1,
        drop_path=0.1,
    )
