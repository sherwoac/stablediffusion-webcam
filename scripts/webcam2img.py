"""make variations of input image"""

import argparse, os
import PIL
import torch
import numpy as np
import cv2
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch import autocast
from contextlib import nullcontext
from pytorch_lightning import seed_everything
from imwatermark import WatermarkEncoder


from scripts.txt2img import put_watermark
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def common_load_image(image: PIL.Image) -> torch.Tensor:
    w, h = image.size
    print(f"loaded input image of size ({w}, {h}) from webcam")
    w, h = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 64
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2. * image - 1.


class CaptureWebcam(object):
    def __init__(self, which_cam: int = 0):
        self._capture = cv2.VideoCapture(which_cam)

    def __del__(self):
        self._capture.release()

    def get_image(self):
        ret, image = self._capture.read()
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def display_arrays(arrays: list):
    arrays = torch.stack(arrays).permute(0, 3, 1, 2)
    grid = make_grid(arrays, nrow=3)
    grid = (rearrange(grid, 'c h w -> h w c').cpu().numpy())
    grid = cv2.cvtColor(grid, cv2.COLOR_BGR2RGB)
    cv2.imshow("output", grid)
    cv2.pollKey()


def load_image_webcam(wc: CaptureWebcam) -> PIL.Image:
    return common_load_image(Image.fromarray(wc.get_image()))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )

    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )

    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across all samples ",
    )

    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )

    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor, most often 8 or 16",
    )

    # parser.add_argument(
    #     "--n_samples",
    #     type=int,
    #     default=2,
    #     help="how many samples to produce for each given prompt. A.k.a batch size",
    # )

    # parser.add_argument(
    #     "--n_rows",
    #     type=int,
    #     default=0,
    #     help="rows in the grid (default: n_samples)",
    # )

    parser.add_argument(
        "--scale",
        type=float,
        default=9.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )

    parser.add_argument(
        "--strength",
        type=float,
        default=0.8,
        help="strength for noising/unnoising. 1.0 corresponds to full destruction of information in init image",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v2-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    return parser.parse_args()


def main():
    opt = parse_args()
    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    sampler = DDIMSampler(model)

    assert 0. <= opt.strength <= 1., 'can only work with strength in [0.0, 1.0]'
    prompt = opt.prompt
    assert prompt is not None
    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    uc = None
    if opt.scale != 1.0:
        uc = model.get_learned_conditioning([""])

    wc = CaptureWebcam()

    sampler.make_schedule(ddim_num_steps=opt.ddim_steps, ddim_eta=opt.ddim_eta, verbose=False)
    c = model.get_learned_conditioning([prompt])
    while True:
        init_image = load_image_webcam(wc).to(device)
        init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space

        t_enc = int(opt.strength * opt.ddim_steps)

        with torch.no_grad():
            with precision_scope("cuda"):
                with model.ema_scope():
                    for _ in trange(opt.n_iter, desc="Sampling"):
                        # encode (scaled latent)
                        z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]).to(device))
                        # decode it
                        samples = sampler.decode(z_enc,
                                                 c,
                                                 t_enc,
                                                 unconditional_guidance_scale=opt.scale,
                                                 unconditional_conditioning=uc)

                        x_samples = model.decode_first_stage(samples)
                        x_sample = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0).squeeze()
                        display_arrays([rearrange(init_image.squeeze().cpu(), 'c h w -> h w c'),
                                        rearrange(x_sample.cpu(), 'c h w -> h w c')])


if __name__ == "__main__":
    main()
