import torch


def load_from_standard_weights(input_file: str, device: str) -> dict[str, torch.Tensor]:
    original_model = torch.load(input_file, map_location=device, weights_only=False)[
        "state_dict"
    ]

    converted = {}
    converted["encoder"] = {}

    converted["encoder"]["0.weight"] = original_model[
        "first_stage_model.encoder.conv_in.weight"
    ]
    converted["encoder"]["0.bias"] = original_model[
        "first_stage_model.encoder.conv_in.bias"
    ]
    converted["encoder"]["1.groupnorm_1.weight"] = original_model[
        "first_stage_model.encoder.down.0.block.0.norm1.weight"
    ]
    converted["encoder"]["1.groupnorm_1.bias"] = original_model[
        "first_stage_model.encoder.down.0.block.0.norm1.bias"
    ]
    converted["encoder"]["1.conv_1.weight"] = original_model[
        "first_stage_model.encoder.down.0.block.0.conv1.weight"
    ]
    converted["encoder"]["1.conv_1.bias"] = original_model[
        "first_stage_model.encoder.down.0.block.0.conv1.bias"
    ]
    converted["encoder"]["1.groupnorm_2.weight"] = original_model[
        "first_stage_model.encoder.down.0.block.0.norm2.weight"
    ]
    converted["encoder"]["1.groupnorm_2.bias"] = original_model[
        "first_stage_model.encoder.down.0.block.0.norm2.bias"
    ]
    converted["encoder"]["1.conv_2.weight"] = original_model[
        "first_stage_model.encoder.down.0.block.0.conv2.weight"
    ]
    converted["encoder"]["1.conv_2.bias"] = original_model[
        "first_stage_model.encoder.down.0.block.0.conv2.bias"
    ]
    converted["encoder"]["2.groupnorm_1.weight"] = original_model[
        "first_stage_model.encoder.down.0.block.1.norm1.weight"
    ]
    converted["encoder"]["2.groupnorm_1.bias"] = original_model[
        "first_stage_model.encoder.down.0.block.1.norm1.bias"
    ]
    converted["encoder"]["2.conv_1.weight"] = original_model[
        "first_stage_model.encoder.down.0.block.1.conv1.weight"
    ]
    converted["encoder"]["2.conv_1.bias"] = original_model[
        "first_stage_model.encoder.down.0.block.1.conv1.bias"
    ]
    converted["encoder"]["2.groupnorm_2.weight"] = original_model[
        "first_stage_model.encoder.down.0.block.1.norm2.weight"
    ]
    converted["encoder"]["2.groupnorm_2.bias"] = original_model[
        "first_stage_model.encoder.down.0.block.1.norm2.bias"
    ]
    converted["encoder"]["2.conv_2.weight"] = original_model[
        "first_stage_model.encoder.down.0.block.1.conv2.weight"
    ]
    converted["encoder"]["2.conv_2.bias"] = original_model[
        "first_stage_model.encoder.down.0.block.1.conv2.bias"
    ]
    converted["encoder"]["3.weight"] = original_model[
        "first_stage_model.encoder.down.0.downsample.conv.weight"
    ]
    converted["encoder"]["3.bias"] = original_model[
        "first_stage_model.encoder.down.0.downsample.conv.bias"
    ]
    converted["encoder"]["4.groupnorm_1.weight"] = original_model[
        "first_stage_model.encoder.down.1.block.0.norm1.weight"
    ]
    converted["encoder"]["4.groupnorm_1.bias"] = original_model[
        "first_stage_model.encoder.down.1.block.0.norm1.bias"
    ]
    converted["encoder"]["4.conv_1.weight"] = original_model[
        "first_stage_model.encoder.down.1.block.0.conv1.weight"
    ]
    converted["encoder"]["4.conv_1.bias"] = original_model[
        "first_stage_model.encoder.down.1.block.0.conv1.bias"
    ]
    converted["encoder"]["4.groupnorm_2.weight"] = original_model[
        "first_stage_model.encoder.down.1.block.0.norm2.weight"
    ]
    converted["encoder"]["4.groupnorm_2.bias"] = original_model[
        "first_stage_model.encoder.down.1.block.0.norm2.bias"
    ]
    converted["encoder"]["4.conv_2.weight"] = original_model[
        "first_stage_model.encoder.down.1.block.0.conv2.weight"
    ]
    converted["encoder"]["4.conv_2.bias"] = original_model[
        "first_stage_model.encoder.down.1.block.0.conv2.bias"
    ]
    converted["encoder"]["4.residual_layer.weight"] = original_model[
        "first_stage_model.encoder.down.1.block.0.nin_shortcut.weight"
    ]
    converted["encoder"]["4.residual_layer.bias"] = original_model[
        "first_stage_model.encoder.down.1.block.0.nin_shortcut.bias"
    ]
    converted["encoder"]["5.groupnorm_1.weight"] = original_model[
        "first_stage_model.encoder.down.1.block.1.norm1.weight"
    ]
    converted["encoder"]["5.groupnorm_1.bias"] = original_model[
        "first_stage_model.encoder.down.1.block.1.norm1.bias"
    ]
    converted["encoder"]["5.conv_1.weight"] = original_model[
        "first_stage_model.encoder.down.1.block.1.conv1.weight"
    ]
    converted["encoder"]["5.conv_1.bias"] = original_model[
        "first_stage_model.encoder.down.1.block.1.conv1.bias"
    ]
    converted["encoder"]["5.groupnorm_2.weight"] = original_model[
        "first_stage_model.encoder.down.1.block.1.norm2.weight"
    ]
    converted["encoder"]["5.groupnorm_2.bias"] = original_model[
        "first_stage_model.encoder.down.1.block.1.norm2.bias"
    ]
    converted["encoder"]["5.conv_2.weight"] = original_model[
        "first_stage_model.encoder.down.1.block.1.conv2.weight"
    ]
    converted["encoder"]["5.conv_2.bias"] = original_model[
        "first_stage_model.encoder.down.1.block.1.conv2.bias"
    ]
    converted["encoder"]["6.weight"] = original_model[
        "first_stage_model.encoder.down.1.downsample.conv.weight"
    ]
    converted["encoder"]["6.bias"] = original_model[
        "first_stage_model.encoder.down.1.downsample.conv.bias"
    ]
    converted["encoder"]["7.groupnorm_1.weight"] = original_model[
        "first_stage_model.encoder.down.2.block.0.norm1.weight"
    ]
    converted["encoder"]["7.groupnorm_1.bias"] = original_model[
        "first_stage_model.encoder.down.2.block.0.norm1.bias"
    ]
    converted["encoder"]["7.conv_1.weight"] = original_model[
        "first_stage_model.encoder.down.2.block.0.conv1.weight"
    ]
    converted["encoder"]["7.conv_1.bias"] = original_model[
        "first_stage_model.encoder.down.2.block.0.conv1.bias"
    ]
    converted["encoder"]["7.groupnorm_2.weight"] = original_model[
        "first_stage_model.encoder.down.2.block.0.norm2.weight"
    ]
    converted["encoder"]["7.groupnorm_2.bias"] = original_model[
        "first_stage_model.encoder.down.2.block.0.norm2.bias"
    ]
    converted["encoder"]["7.conv_2.weight"] = original_model[
        "first_stage_model.encoder.down.2.block.0.conv2.weight"
    ]
    converted["encoder"]["7.conv_2.bias"] = original_model[
        "first_stage_model.encoder.down.2.block.0.conv2.bias"
    ]
    converted["encoder"]["7.residual_layer.weight"] = original_model[
        "first_stage_model.encoder.down.2.block.0.nin_shortcut.weight"
    ]
    converted["encoder"]["7.residual_layer.bias"] = original_model[
        "first_stage_model.encoder.down.2.block.0.nin_shortcut.bias"
    ]
    converted["encoder"]["8.groupnorm_1.weight"] = original_model[
        "first_stage_model.encoder.down.2.block.1.norm1.weight"
    ]
    converted["encoder"]["8.groupnorm_1.bias"] = original_model[
        "first_stage_model.encoder.down.2.block.1.norm1.bias"
    ]
    converted["encoder"]["8.conv_1.weight"] = original_model[
        "first_stage_model.encoder.down.2.block.1.conv1.weight"
    ]
    converted["encoder"]["8.conv_1.bias"] = original_model[
        "first_stage_model.encoder.down.2.block.1.conv1.bias"
    ]
    converted["encoder"]["8.groupnorm_2.weight"] = original_model[
        "first_stage_model.encoder.down.2.block.1.norm2.weight"
    ]
    converted["encoder"]["8.groupnorm_2.bias"] = original_model[
        "first_stage_model.encoder.down.2.block.1.norm2.bias"
    ]
    converted["encoder"]["8.conv_2.weight"] = original_model[
        "first_stage_model.encoder.down.2.block.1.conv2.weight"
    ]
    converted["encoder"]["8.conv_2.bias"] = original_model[
        "first_stage_model.encoder.down.2.block.1.conv2.bias"
    ]
    converted["encoder"]["9.weight"] = original_model[
        "first_stage_model.encoder.down.2.downsample.conv.weight"
    ]
    converted["encoder"]["9.bias"] = original_model[
        "first_stage_model.encoder.down.2.downsample.conv.bias"
    ]
    converted["encoder"]["10.groupnorm_1.weight"] = original_model[
        "first_stage_model.encoder.down.3.block.0.norm1.weight"
    ]
    converted["encoder"]["10.groupnorm_1.bias"] = original_model[
        "first_stage_model.encoder.down.3.block.0.norm1.bias"
    ]
    converted["encoder"]["10.conv_1.weight"] = original_model[
        "first_stage_model.encoder.down.3.block.0.conv1.weight"
    ]
    converted["encoder"]["10.conv_1.bias"] = original_model[
        "first_stage_model.encoder.down.3.block.0.conv1.bias"
    ]
    converted["encoder"]["10.groupnorm_2.weight"] = original_model[
        "first_stage_model.encoder.down.3.block.0.norm2.weight"
    ]
    converted["encoder"]["10.groupnorm_2.bias"] = original_model[
        "first_stage_model.encoder.down.3.block.0.norm2.bias"
    ]
    converted["encoder"]["10.conv_2.weight"] = original_model[
        "first_stage_model.encoder.down.3.block.0.conv2.weight"
    ]
    converted["encoder"]["10.conv_2.bias"] = original_model[
        "first_stage_model.encoder.down.3.block.0.conv2.bias"
    ]
    converted["encoder"]["11.groupnorm_1.weight"] = original_model[
        "first_stage_model.encoder.down.3.block.1.norm1.weight"
    ]
    converted["encoder"]["11.groupnorm_1.bias"] = original_model[
        "first_stage_model.encoder.down.3.block.1.norm1.bias"
    ]
    converted["encoder"]["11.conv_1.weight"] = original_model[
        "first_stage_model.encoder.down.3.block.1.conv1.weight"
    ]
    converted["encoder"]["11.conv_1.bias"] = original_model[
        "first_stage_model.encoder.down.3.block.1.conv1.bias"
    ]
    converted["encoder"]["11.groupnorm_2.weight"] = original_model[
        "first_stage_model.encoder.down.3.block.1.norm2.weight"
    ]
    converted["encoder"]["11.groupnorm_2.bias"] = original_model[
        "first_stage_model.encoder.down.3.block.1.norm2.bias"
    ]
    converted["encoder"]["11.conv_2.weight"] = original_model[
        "first_stage_model.encoder.down.3.block.1.conv2.weight"
    ]
    converted["encoder"]["11.conv_2.bias"] = original_model[
        "first_stage_model.encoder.down.3.block.1.conv2.bias"
    ]
    converted["encoder"]["12.groupnorm_1.weight"] = original_model[
        "first_stage_model.encoder.mid.block_1.norm1.weight"
    ]
    converted["encoder"]["12.groupnorm_1.bias"] = original_model[
        "first_stage_model.encoder.mid.block_1.norm1.bias"
    ]
    converted["encoder"]["12.conv_1.weight"] = original_model[
        "first_stage_model.encoder.mid.block_1.conv1.weight"
    ]
    converted["encoder"]["12.conv_1.bias"] = original_model[
        "first_stage_model.encoder.mid.block_1.conv1.bias"
    ]
    converted["encoder"]["12.groupnorm_2.weight"] = original_model[
        "first_stage_model.encoder.mid.block_1.norm2.weight"
    ]
    converted["encoder"]["12.groupnorm_2.bias"] = original_model[
        "first_stage_model.encoder.mid.block_1.norm2.bias"
    ]
    converted["encoder"]["12.conv_2.weight"] = original_model[
        "first_stage_model.encoder.mid.block_1.conv2.weight"
    ]
    converted["encoder"]["12.conv_2.bias"] = original_model[
        "first_stage_model.encoder.mid.block_1.conv2.bias"
    ]
    converted["encoder"]["13.groupnorm.weight"] = original_model[
        "first_stage_model.encoder.mid.attn_1.norm.weight"
    ]
    converted["encoder"]["13.groupnorm.bias"] = original_model[
        "first_stage_model.encoder.mid.attn_1.norm.bias"
    ]
    converted["encoder"]["13.attention.out_proj.bias"] = original_model[
        "first_stage_model.encoder.mid.attn_1.proj_out.bias"
    ]
    converted["encoder"]["14.groupnorm_1.weight"] = original_model[
        "first_stage_model.encoder.mid.block_2.norm1.weight"
    ]
    converted["encoder"]["14.groupnorm_1.bias"] = original_model[
        "first_stage_model.encoder.mid.block_2.norm1.bias"
    ]
    converted["encoder"]["14.conv_1.weight"] = original_model[
        "first_stage_model.encoder.mid.block_2.conv1.weight"
    ]
    converted["encoder"]["14.conv_1.bias"] = original_model[
        "first_stage_model.encoder.mid.block_2.conv1.bias"
    ]
    converted["encoder"]["14.groupnorm_2.weight"] = original_model[
        "first_stage_model.encoder.mid.block_2.norm2.weight"
    ]
    converted["encoder"]["14.groupnorm_2.bias"] = original_model[
        "first_stage_model.encoder.mid.block_2.norm2.bias"
    ]
    converted["encoder"]["14.conv_2.weight"] = original_model[
        "first_stage_model.encoder.mid.block_2.conv2.weight"
    ]
    converted["encoder"]["14.conv_2.bias"] = original_model[
        "first_stage_model.encoder.mid.block_2.conv2.bias"
    ]
    converted["encoder"]["15.weight"] = original_model[
        "first_stage_model.encoder.norm_out.weight"
    ]
    converted["encoder"]["15.bias"] = original_model[
        "first_stage_model.encoder.norm_out.bias"
    ]
    converted["encoder"]["17.weight"] = original_model[
        "first_stage_model.encoder.conv_out.weight"
    ]
    converted["encoder"]["17.bias"] = original_model[
        "first_stage_model.encoder.conv_out.bias"
    ]
    converted["encoder"]["18.weight"] = original_model[
        "first_stage_model.quant_conv.weight"
    ]
    converted["encoder"]["18.bias"] = original_model[
        "first_stage_model.quant_conv.bias"
    ]
    converted["encoder"]["13.attention.in_proj.weight"] = torch.cat(
        (
            original_model["first_stage_model.encoder.mid.attn_1.q.weight"],
            original_model["first_stage_model.encoder.mid.attn_1.k.weight"],
            original_model["first_stage_model.encoder.mid.attn_1.v.weight"],
        ),
        0,
    ).reshape((1536, 512))
    converted["encoder"]["13.attention.in_proj.bias"] = torch.cat(
        (
            original_model["first_stage_model.encoder.mid.attn_1.q.bias"],
            original_model["first_stage_model.encoder.mid.attn_1.k.bias"],
            original_model["first_stage_model.encoder.mid.attn_1.v.bias"],
        ),
        0,
    )
    converted["encoder"]["13.attention.out_proj.weight"] = original_model[
        "first_stage_model.encoder.mid.attn_1.proj_out.weight"
    ].reshape((512, 512))

    return converted
