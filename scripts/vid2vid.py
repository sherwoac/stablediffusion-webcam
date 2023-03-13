"""make variations of input image"""

import argparse, os
import PIL
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch import autocast
from contextlib import nullcontext
from pytorch_lightning import seed_everything

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

import moviepy.editor


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


def load_img(path):
    image = Image.open(path).convert("RGB")
    print(f"loaded input image from {path}")
    return convert_frame_image(image=image)


def resize_image(image: np.ndarray) -> np.ndarray:
    h, w, c = image.shape
    w, h = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 64
    pil_image = Image.fromarray(image).resize((w, h), resample=PIL.Image.LANCZOS)
    return np.array(pil_image)


def convert_frame_image(image: np.ndarray):
    image = resize_image(image)
    image = image.astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2) # h, w, c -> 0, c, h, w
    image = torch.from_numpy(image)
    return 2. * image - 1.


def load_vid(path: str) -> moviepy.editor.VideoFileClip:
    return moviepy.editor.VideoFileClip(path)


def tensor_to_image(image_tensor: torch.Tensor) -> PIL.Image:
    image_tensor = torch.clamp((image_tensor + 1.0) / 2.0, min=0.0, max=1.0)
    int_image_tensor = (255. * image_tensor.permute(0, 2, 3, 1)).to(torch.uint8)
    return Image.fromarray(int_image_tensor[0].cpu().numpy())


def save_image_tensor(image_tensor: torch.Tensor, path: str, count: int):
    img = tensor_to_image(image_tensor=image_tensor)
    img.save(os.path.join(path, f"{count:08d}.png"))


def get_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )

    parser.add_argument(
        "--inputvid",
        type=str,
        nargs="?",
        help="path to the input vid"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        nargs="?",
        help="output_dir to write results to",
        default="./outputs/"
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

    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )

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
        "--subtitles",
        type=str,
        help="if specified, load prompts from this subtitle file",
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

    parser.add_argument(
        "--skip_frame",
        type=int,
        default=38*30,
        help="the seed (for reproducible sampling)",
    )


    parser.add_argument(
        "--random_each_frame",
        action='store_true',
        default=False,
        help="like corridor crew: https://youtu.be/_9LX9HSQkWo?t=188 "
    )

    parser.add_argument(
        "--generate_random_from_image",
        action='store_true',
        default=False,
        help="like corridor crew: https://youtu.be/_9LX9HSQkWo?t=215 "
    )

    return parser


def main():
    opt = get_arg_parser().parse_args()
    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    sampler = DDIMSampler(model)

    os.makedirs(opt.output_dir, exist_ok=True)
    
    if not opt.subtitles:
        prompt = opt.prompt
        assert prompt is not None
        prompt_data = opt.prompt

    else:
        raise NotImplementedError()

    assert os.path.isfile(opt.inputvid), f'file not found at: {opt.inputvid=}'
    vid = load_vid(opt.inputvid)
 
    sampler.make_schedule(ddim_num_steps=opt.ddim_steps, ddim_eta=opt.ddim_eta, verbose=False)

    assert 0. <= opt.strength <= 1., 'can only work with strength in [0.0, 1.0]'
    t_enc = int(opt.strength * opt.ddim_steps)
    print(f"target t_enc is {t_enc} steps")

    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    
    # fixed noise
    init_frame = vid.make_frame(0.)
    init_image = convert_frame_image(init_frame).to(device)
    init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))
    latent_noise = torch.randn_like(init_latent)

    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                all_samples = list()
                for frame_number, video_frame in enumerate(vid.iter_frames()):
                    
                    # skip some (intro?) frames
                    if frame_number < opt.skip_frame:
                        continue

                    uc = None
                    if opt.scale != 1.0:
                        uc = model.get_learned_conditioning([""])

                    c = model.get_learned_conditioning(prompt_data)

                    init_image = convert_frame_image(video_frame).to(device)
                    init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image)) 

                    if opt.random_each_frame:
                        latent_noise = torch.randn_like(init_latent)

                    if opt.generate_random_from_image:
                        sum_colour = init_image.permute(0, 2, 3, 1).sum(dim=3).unsqueeze(dim=-1).repeat_interleave(3, dim=3).permute(0, 3, 1, 2)
                        sum_colour = (sum_colour - sum_colour.mean()) / sum_colour.std()
                        latent_noise = model.get_first_stage_encoding(model.encode_first_stage(sum_colour)) 

                    # encode (scaled latent)
                    z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]).to(device), noise=latent_noise)

                    # decode it
                    sample = sampler.decode(z_enc, 
                                             c, 
                                             t_enc, 
                                             unconditional_guidance_scale=opt.scale,
                                             unconditional_conditioning=uc,
                                             repeat_noise=False)

                    x_sample = model.decode_first_stage(sample)
                    save_image_tensor(x_sample, opt.output_dir, frame_number) 


    print(f"Your samples are ready and waiting for you here: \n{os.path.dirname(opt.outfile)} \nEnjoy.")


if __name__ == "__main__":
    main()
