import argparse
import numpy as np

from improved_diffusion import dist_util
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from ood_utils import load_data, dict2namespace
from tqdm import tqdm
import yaml
import os


def main():

    args = create_argparser().parse_args()
    dist_util.setup_dist(args.device)
    args.timestep_respacing = f"ddim{args.n_ddim_steps}"
    
    with open(args.config, 'r') as fp:
        config = yaml.safe_load(fp)
    config = dict2namespace(config)

    # merge config into args
    for arg_name, arg_value in vars(config).items():
        setattr(args, arg_name, arg_value)

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    print(f"loading model from {args.model_path}")
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    print(f"Loading {args.dataset}")
    dataloader = load_data(args.dataset, args.data_dir, args.batch_size, args.image_size, train=True)
    reverse_sample_fn = diffusion.ddim_reverse_sample_loop
    n_ddim_steps = len(diffusion.betas)
    
    # statistics of diffusion path to save
    eps_sum_arr = [] # sum of eps
    eps_sum_abs_arr = [] # sum of eps absolute val
    eps_sum_sq_arr = [] # sum of eps squared
    eps_sum_sq_sqrt_arr = [] # sum of eps squared square root
    eps_sum_cb_arr = [] # sum of eps cubed
    eps_sum_cb_cbrt_arr = [] # sum of eps cubed cube root
    
    deps_dt_arr = [] # sum of rate-of-change deps/dt
    deps_dt_abs_arr = [] # sum of deps/dt absolute val
    deps_dt_sq_arr = [] # sum of deps/dt squared
    deps_dt_sq_sqrt_arr = [] # sum of deps/dt squared square root
    deps_dt_cb_arr = [] # sum of deps/dt cubed 
    deps_dt_cb_cbrt_arr = [] # sum of deps/dt cubed cube root
    
    for data in tqdm(dataloader, f"encoding {args.dataset} with {args.timestep_respacing} and calculating all statistics"):
        x0 = data[0].to(dist_util.dev())
        _, eps = reverse_sample_fn(
            model,
            x0.shape,
            x0,
            clip_denoised=args.clip_denoised,
            model_kwargs=None,
            return_eps=True,
            return_xt=False
        )
        eps = [x.cpu().numpy() for x in eps]
        eps = np.array(eps).transpose(1,0,2,3,4) # B, T, C, H, W
        eps_sum = np.sum(eps, axis=(1,2,3,4))
        eps_sum_abs = np.sum(np.abs(eps), axis=(1,2,3,4))
        eps_sum_sq = np.sum(eps**2, axis=(1,2,3,4))
        eps_sum_sq_sqrt = np.sqrt(np.sum(eps**2, axis=(1,2,3,4)))
        eps_sum_cb = np.sum(eps**3, axis=(1,2,3,4))
        eps_sum_cb_cbrt = np.cbrt(np.sum(eps**3, axis=(1,2,3,4)))
        eps_sum_arr.extend(eps_sum.tolist())
        eps_sum_abs_arr.extend(eps_sum_abs.tolist())
        eps_sum_sq_arr.extend(eps_sum_sq.tolist())
        eps_sum_sq_sqrt_arr.extend(eps_sum_sq_sqrt.tolist())
        eps_sum_cb_arr.extend(eps_sum_cb.tolist())
        eps_sum_cb_cbrt_arr.extend(eps_sum_cb_cbrt.tolist())

        eps_diff = np.diff(eps, axis=1) * n_ddim_steps # ep 5
        deps_dt = np.sum(eps_diff, axis=(1,2,3,4))
        deps_dt_abs = np.sum(np.abs(eps_diff), axis=(1,2,3,4))
        deps_dt_sq = np.sum(eps_diff**2, axis=(1,2,3,4))
        deps_dt_sq_sqrt = np.sqrt(np.sum(eps_diff**2, axis=(1,2,3,4)))
        deps_dt_cb = np.sum(eps_diff**3, axis=(1,2,3,4))
        deps_dt_cb_cbrt = np.cbrt(np.sum(eps_diff**3, axis=(1,2,3,4)))
        deps_dt_arr.extend(deps_dt.tolist())
        deps_dt_abs_arr.extend(deps_dt_abs.tolist())
        deps_dt_sq_arr.extend(deps_dt_sq.tolist())
        deps_dt_sq_sqrt_arr.extend(deps_dt_sq_sqrt.tolist())
        deps_dt_cb_arr.extend(deps_dt_cb.tolist())
        deps_dt_cb_cbrt_arr.extend(deps_dt_cb_cbrt.tolist())
    
    eps_sum_arr = np.array(eps_sum_arr)
    eps_sum_abs_arr = np.array(eps_sum_abs_arr)
    eps_sum_sq_arr = np.array(eps_sum_sq_arr)
    eps_sum_sq_sqrt_arr = np.array(eps_sum_sq_sqrt_arr)
    eps_sum_cb_arr = np.array(eps_sum_cb_arr)
    eps_sum_cb_cbrt_arr = np.array(eps_sum_cb_cbrt_arr)
    deps_dt_arr = np.array(deps_dt_arr)
    deps_dt_abs_arr = np.array(deps_dt_abs_arr)
    deps_dt_sq_arr = np.array(deps_dt_sq_arr)
    deps_dt_sq_sqrt_arr = np.array(deps_dt_sq_sqrt_arr)
    deps_dt_cb_arr = np.array(deps_dt_cb_arr)
    deps_dt_cb_cbrt_arr = np.array(deps_dt_cb_cbrt_arr)

    save_dir = f"train_statistics/{args.timestep_respacing}"
    os.makedirs(save_dir, exist_ok=True)
    np.savez_compressed(os.path.join(save_dir, args.dataset), eps_sum=eps_sum_arr, eps_sum_abs=eps_sum_abs_arr, eps_sum_sq=eps_sum_sq_arr, 
                        eps_sum_sq_sqrt=eps_sum_sq_sqrt_arr, eps_sum_cb=eps_sum_cb_arr, eps_sum_cb_cbrt = eps_sum_cb_cbrt_arr, 
                        deps_dt=deps_dt_arr, deps_dt_abs = deps_dt_abs_arr, deps_dt_sq=deps_dt_sq_arr, 
                        deps_dt_sq_sqrt=deps_dt_sq_sqrt_arr, deps_dt_cb=deps_dt_cb_arr, deps_dt_cb_cbrt=deps_dt_cb_cbrt_arr)


def create_argparser():
    defaults = dict(
        config="configs/imagenet_model_config.yaml",
        batch_size=256,
        n_ddim_steps=50,
        device=0,
        data_dir="",
        dataset="",
        model_path="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()