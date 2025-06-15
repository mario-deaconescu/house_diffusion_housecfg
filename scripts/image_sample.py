"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
# from ctypes.macholib import dyld
# dyld.DEFAULT_LIBRARY_FALLBACK.append("/opt/homebrew/lib")

import argparse
import io
import os

import cv2
import imageio
import torch
import torch as th
import webcolors
from PIL import Image
from matplotlib import pyplot as plt
from tqdm.contrib.concurrent import process_map

from ..house_diffusion import dist_util, logger
from ..house_diffusion.rplanhg_datasets import load_rplanhg_data
from ..house_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
    update_arg_parser,
)
from src.eval.make_eval_gt import save_img_plan
from src.rplan.types import RoomType, TorchTransformerPlan, ImagePlan

# import random
# th.manual_seed(0)
# random.seed(0)
# np.random.seed(0)

bin_to_int = lambda x: int("".join([str(int(i.cpu().data)) for i in x]), 2)


def bin_to_int_sample(sample, resolution=256):
    sample_new = th.zeros([sample.shape[0], sample.shape[1], sample.shape[2], 2])
    sample[sample < 0] = 0
    sample[sample > 0] = 1
    for i in range(sample.shape[0]):
        for j in range(sample.shape[1]):
            for k in range(sample.shape[2]):
                sample_new[i, j, k, 0] = bin_to_int(sample[i, j, k, :8])
                sample_new[i, j, k, 1] = bin_to_int(sample[i, j, k, 8:])
    sample = sample_new
    sample = sample / (resolution / 2) - 1
    return sample


def save_samples(
        sample, ext, model_kwargs,
        tmp_count, num_room_types,
        save_gif=False, save_edges=False,
        door_indices=[11, 12, 13], ID_COLOR=None,
        is_syn=False, draw_graph=False, save_svg=False):
    prefix = 'syn_' if is_syn else ''
    graph_errors = []
    if not save_gif:
        sample = sample[-1:]
    for i in tqdm(range(sample.shape[1])):
        resolution = 256
        images = []
        images2 = []
        images3 = []
        for k in range(sample.shape[0]):
            draw = drawsvg.Drawing(resolution, resolution, displayInline=False)
            draw.append(drawsvg.Rectangle(0, 0, resolution, resolution, fill='black'))
            draw2 = drawsvg.Drawing(resolution, resolution, displayInline=False)
            draw2.append(drawsvg.Rectangle(0, 0, resolution, resolution, fill='black'))
            draw3 = drawsvg.Drawing(resolution, resolution, displayInline=False)
            draw3.append(drawsvg.Rectangle(0, 0, resolution, resolution, fill='black'))
            draw_color = drawsvg.Drawing(resolution, resolution, displayInline=False)
            draw_color.append(drawsvg.Rectangle(0, 0, resolution, resolution, fill='white'))
            polys = []
            types = []
            for j, point in (enumerate(sample[k][i])):
                if model_kwargs[f'{prefix}src_key_padding_mask'][i][j] == 1:
                    continue
                point = point.cpu().data.numpy()
                if j == 0:
                    poly = []
                if j > 0 and (model_kwargs[f'{prefix}room_indices'][i, j] != model_kwargs[f'{prefix}room_indices'][
                    i, j - 1]).any():
                    polys.append(poly)
                    types.append(c)
                    poly = []
                pred_center = False
                if pred_center:
                    point = point / 2 + 1
                    point = point * resolution // 2
                else:
                    point = point / 2 + 0.5
                    point = point * resolution
                poly.append((point[0], point[1]))
                c = np.argmax(model_kwargs[f'{prefix}room_types'][i][j - 1].cpu().numpy())
            polys.append(poly)
            types.append(c)
            for poly, c in zip(polys, types):
                if c in door_indices or c == 0:
                    continue
                room_type = c
                c = webcolors.hex_to_rgb(ID_COLOR[c])
                draw_color.append(
                    drawsvg.Lines(*np.array(poly).flatten().tolist(), close=True, fill=ID_COLOR[room_type],
                                  fill_opacity=1.0, stroke='black', stroke_width=1))
                draw.append(
                    drawsvg.Lines(*np.array(poly).flatten().tolist(), close=True, fill='black', fill_opacity=0.0,
                                  stroke=webcolors.rgb_to_hex([int(x / 2) for x in c]),
                                  stroke_width=0.5 * (resolution / 256)))
                draw2.append(drawsvg.Lines(*np.array(poly).flatten().tolist(), close=True, fill=ID_COLOR[room_type],
                                           fill_opacity=1.0, stroke=webcolors.rgb_to_hex([int(x / 2) for x in c]),
                                           stroke_width=0.5 * (resolution / 256)))
                for corner in poly:
                    draw.append(drawsvg.Circle(corner[0], corner[1], 2 * (resolution / 256), fill=ID_COLOR[room_type],
                                               fill_opacity=1.0, stroke='gray', stroke_width=0.25))
                    draw3.append(drawsvg.Circle(corner[0], corner[1], 2 * (resolution / 256), fill=ID_COLOR[room_type],
                                                fill_opacity=1.0, stroke='gray', stroke_width=0.25))
            for poly, c in zip(polys, types):
                if c not in door_indices:
                    continue
                room_type = c
                c = webcolors.hex_to_rgb(ID_COLOR[c])
                draw_color.append(
                    drawsvg.Lines(*np.array(poly).flatten().tolist(), close=True, fill=ID_COLOR[room_type],
                                  fill_opacity=1.0, stroke='black', stroke_width=1))
                draw.append(
                    drawsvg.Lines(*np.array(poly).flatten().tolist(), close=True, fill='black', fill_opacity=0.0,
                                  stroke=webcolors.rgb_to_hex([int(x / 2) for x in c]),
                                  stroke_width=0.5 * (resolution / 256)))
                draw2.append(drawsvg.Lines(*np.array(poly).flatten().tolist(), close=True, fill=ID_COLOR[room_type],
                                           fill_opacity=1.0, stroke=webcolors.rgb_to_hex([int(x / 2) for x in c]),
                                           stroke_width=0.5 * (resolution / 256)))
                for corner in poly:
                    draw.append(drawsvg.Circle(corner[0], corner[1], 2 * (resolution / 256), fill=ID_COLOR[room_type],
                                               fill_opacity=1.0, stroke='gray', stroke_width=0.25))
                    draw3.append(drawsvg.Circle(corner[0], corner[1], 2 * (resolution / 256), fill=ID_COLOR[room_type],
                                                fill_opacity=1.0, stroke='gray', stroke_width=0.25))
            images.append(Image.open(io.BytesIO(cairosvg.svg2png(draw.asSvg()))))
            images2.append(Image.open(io.BytesIO(cairosvg.svg2png(draw2.asSvg()))))
            images3.append(Image.open(io.BytesIO(cairosvg.svg2png(draw3.asSvg()))))
            if k == sample.shape[0] - 1 or True:
                if save_edges:
                    draw.saveSvg(f'outputs/{ext}/{tmp_count + i}_{k}_{ext}.svg')
                if save_svg:
                    draw_color.saveSvg(f'outputs/{ext}/{tmp_count + i}c_{k}_{ext}.svg')
                else:
                    Image.open(io.BytesIO(cairosvg.svg2png(draw_color.asSvg()))).save(
                        f'outputs/{ext}/{tmp_count + i}c_{ext}.png')
            if k == sample.shape[0] - 1:
                if False and 'graph' in model_kwargs:
                    graph_errors.append(estimate_graph(tmp_count + i, polys, types, model_kwargs[f'{prefix}graph'][i],
                                                       ID_COLOR=ID_COLOR, draw_graph=draw_graph, save_svg=save_svg))
                else:
                    graph_errors.append(0)
        if save_gif:
            imageio.mimwrite(f'outputs/gif/{tmp_count + i}.gif', images, fps=10, loop=1)
            imageio.mimwrite(f'outputs/gif/{tmp_count + i}_v2.gif', images2, fps=10, loop=1)
            imageio.mimwrite(f'outputs/gif/{tmp_count + i}_v3.gif', images3, fps=10, loop=1)
    return graph_errors


def save_plan(sample, model_kwargs, idx: int, previous_count: int, output_path: str):
    global_idx = previous_count + idx
    gt_kwargs = lambda key: model_kwargs[f'syn_{key}']
    room_types = gt_kwargs('room_types')
    room_types_values = [int(value.item()) for value in torch.argmax(room_types, dim=-1)]
    mapping = {11: 15, 12: 17, 13: 16}
    room_types_values = [mapping.get(room_type, room_type) for room_type in room_types_values]
    room_type_enums = [RoomType(room_type) if room_type > 0 else None for room_type in room_types_values]
    room_types = torch.stack(
        [room_type_enum.one_hot(device=room_types.device) if room_type_enum is not None else torch.zeros(
            len(RoomType), device=room_types.device) for room_type_enum in room_type_enums])
    conditions = TorchTransformerPlan.Conditions(
        door_mask=gt_kwargs('door_mask'),
        self_mask=gt_kwargs('self_mask'),
        gen_mask=gt_kwargs('gen_mask'),
        room_types=room_types,
        corner_indices=gt_kwargs('corner_indices'),
        room_indices=gt_kwargs('room_indices'),
        src_key_padding_mask=gt_kwargs('src_key_padding_mask') > 0,
        connections=gt_kwargs('connections'),
    )
    plan = TorchTransformerPlan(coordinates=sample, conditions=conditions).to_plan()

    image_plan = ImagePlan.from_plan(plan, mask_size=256)
    img = image_plan.to_image()
    cv2.imwrite(os.path.join(output_path, f"{global_idx}.png"), img)


def main():
    args = create_argparser().parse_args()
    update_arg_parser(args)

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        torch.load(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    logger.log("sampling...")
    tmp_count = 0
    output_path = args.output_path
    assert output_path is not None, "Output path must be specified"
    os.makedirs(output_path, exist_ok=True)

    data = load_rplanhg_data(
        batch_size=args.batch_size,
        analog_bit=args.analog_bit,
        set_name=args.set_name,
        target_set=args.target_set,
    )
    while tmp_count < args.num_samples:
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        data_sample, model_kwargs = next(data)
        for key in model_kwargs:
            model_kwargs[key] = model_kwargs[key].float().to(dist_util.dev())

        samples = sample_fn(
            model,
            data_sample.shape,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            analog_bit=args.analog_bit,
        )
        # sample_gt = data_sample.cpu().unsqueeze(0)
        samples = samples.permute([0, 1, 3, 2])
        # sample_gt = sample_gt.permute([0, 1, 3, 2])
        if args.analog_bit:
            # sample_gt = bin_to_int_sample(sample_gt)
            samples = bin_to_int_sample(samples)

        samples_step = samples[-1].detach().cpu()
        # samples_step = [samples_step[i] for i in range(samples_step.shape[0])]
        kwargs_list = []
        for i in range(len(samples_step)):
            kwargs = {}
            for key in model_kwargs:
                kwargs[key] = model_kwargs[key][i].detach().cpu()
            kwargs_list.append(kwargs)

        process_map(save_plan, samples_step, kwargs_list, range(len(samples_step)),
                    [tmp_count] * len(samples_step), [output_path] * len(samples_step),
                    max_workers=min(10, len(samples_step)),
                    desc=f"Generating: {tmp_count}/{args.num_samples}",
                    total=len(samples_step))
        tmp_count += len(samples_step)
        # plan.visualize(view_room_type=False, view_door_index=False, view_graph=False)
        # plt.show()


def create_argparser():
    defaults = dict(
        dataset='',
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        draw_graph=True,
        save_svg=True,
        output_path="outputs/house_diffusion/",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
