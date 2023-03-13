import torch
import numpy as np
import k_diffusion as K

from PIL import Image
from torch import autocast
from einops import rearrange, repeat

def pil_img_to_torch(pil_img, half=False):
    image = np.array(pil_img).astype(np.float32) / 255.0
    image = rearrange(torch.from_numpy(image), 'h w c -> c h w')
    if half:
        image = image.half()
    return (2.0 * image - 1.0).unsqueeze(0)

def pil_img_to_latent(model, img, batch_size=1, device='cuda', half=False):
    init_image = pil_img_to_torch(img, half=half).to(device)
    init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
    if half:
        return model.get_first_stage_encoding(model.encode_first_stage(init_image.half()))
    return model.get_first_stage_encoding(model.encode_first_stage(init_image))

def find_noise_for_image(sd_model, img, prompt, steps=200, cond_scale=0.0, verbose=False, normalize=True, convert_image=False):
    if convert_image:
        x = pil_img_to_latent(sd_model, img, batch_size=1, device='cuda', half=True)
    else:
        x = sd_model.get_first_stage_encoding(sd_model.encode_first_stage(img))

    with torch.no_grad():
        with autocast('cuda'):
            uncond = sd_model.get_learned_conditioning([''])
            cond = sd_model.get_learned_conditioning([prompt])

    s_in = x.new_ones([x.shape[0]])
    dnw = K.external.CompVisDenoiser(sd_model)
    sigmas = dnw.get_sigmas(steps).flip(0)

    if verbose:
        print(sigmas)

    with torch.no_grad():
        with autocast('cuda'):
            for i in range(1, len(sigmas)):
                x_in = torch.cat([x] * 2)
                sigma_in = torch.cat([sigmas[i - 1] * s_in] * 2)
                cond_in = torch.cat([uncond, cond])

                c_out, c_in = [K.utils.append_dims(k, x_in.ndim) for k in dnw.get_scalings(sigma_in)]
                
                if i == 1:
                    t = dnw.sigma_to_t(torch.cat([sigmas[i] * s_in] * 2))
                else:
                    t = dnw.sigma_to_t(sigma_in)
                    
                eps = sd_model.apply_model(x_in * c_in, t, cond=cond_in)
                denoised_uncond, denoised_cond = (x_in + eps * c_out).chunk(2)
                
                denoised = denoised_uncond + (denoised_cond - denoised_uncond) * cond_scale
                
                if i == 1:
                    d = (x - denoised) / (2 * sigmas[i])
                else:
                    d = (x - denoised) / sigmas[i - 1]

                dt = sigmas[i] - sigmas[i - 1]
                x = x + d * dt
            
            if normalize:
                return x / x.std()
            else:
                return x