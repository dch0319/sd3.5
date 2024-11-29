import math
import os

import numpy as np
import torch
from safetensors import safe_open
from tqdm import tqdm

from other_impls import SD3Tokenizer, T5XXLModel, SDXLClipG, SDClipModel
from sd3_impls import SkipLayerCFGDenoiser, CFGDenoiser, SD3LatentFormat, SDVAE, BaseModel


#################################################################################################
### Wrappers for model parts
#################################################################################################


def load_into(f, model, prefix, device, dtype=None):
    """Just a debugging-friendly hack to apply the weights in a safetensors file to the pytorch module."""
    for key in f.keys():
        if key.startswith(prefix) and not key.startswith("loss."):
            path = key[len(prefix):].split(".")
            obj = model
            for p in path:
                if obj is list:
                    obj = obj[int(p)]
                else:
                    obj = getattr(obj, p, None)
                    if obj is None:
                        print(
                            f"Skipping key '{key}' in safetensors file as '{p}' does not exist in python model"
                        )
                        break
            if obj is None:
                continue
            try:
                tensor = f.get_tensor(key).to(device=device)
                if dtype is not None:
                    tensor = tensor.to(dtype=dtype)
                obj.requires_grad_(False)
                obj.set_(tensor)
            except Exception as e:
                print(f"Failed to load key '{key}' in safetensors file: {e}")
                raise e


CLIPG_CONFIG = {
    "hidden_act": "gelu",
    "hidden_size": 1280,
    "intermediate_size": 5120,
    "num_attention_heads": 20,
    "num_hidden_layers": 32,
}


class ClipG:
    def __init__(self):
        with safe_open("models/clip_g.safetensors", framework="pt", device="cpu") as f:
            self.model = SDXLClipG(CLIPG_CONFIG, device="cpu", dtype=torch.float32)
            load_into(f, self.model.transformer, "", "cpu", torch.float32)


CLIPL_CONFIG = {
    "hidden_act": "quick_gelu",
    "hidden_size": 768,
    "intermediate_size": 3072,
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
}


class ClipL:
    def __init__(self):
        with safe_open("models/clip_l.safetensors", framework="pt", device="cpu") as f:
            self.model = SDClipModel(
                layer="hidden",
                layer_idx=-2,
                device="cpu",
                dtype=torch.float32,
                layer_norm_hidden_state=False,
                return_projected_pooled=False,
                textmodel_json_config=CLIPL_CONFIG,
            )
            load_into(f, self.model.transformer, "", "cpu", torch.float32)


T5_CONFIG = {
    "d_ff": 10240,
    "d_model": 4096,
    "num_heads": 64,
    "num_layers": 24,
    "vocab_size": 32128,
}


class T5XXL:
    def __init__(self):
        with safe_open("models/t5xxl_fp16.safetensors", framework="pt", device="cpu") as f:
            self.model = T5XXLModel(T5_CONFIG, device="cpu", dtype=torch.float32)
            load_into(f, self.model.transformer, "", "cpu", torch.float32)


class SD3:
    def __init__(self, model, shift, verbose=False):
        with safe_open(model, framework="pt", device="cpu") as f:
            self.model = BaseModel(
                shift=shift,
                file=f,
                prefix="model.diffusion_model.",
                device="cpu",
                dtype=torch.float16,
                verbose=verbose,
            ).eval()
            load_into(f, self.model, "model.", "cpu", torch.float16)


class VAE:
    def __init__(self, model):
        with safe_open(model, framework="pt", device="cpu") as f:
            self.model = SDVAE(device="cpu", dtype=torch.float16).eval().cpu()
            prefix = ""
            if any(k.startswith("first_stage_model.") for k in f.keys()):
                prefix = "first_stage_model."
            load_into(f, self.model, prefix, "cpu", torch.float16)


# @torch.no_grad()
@torch.autocast("cuda", dtype=torch.float16)
def sample_dpmpp_2m(model, x, sigmas, extra_args=None):
    """DPM-Solver++(2M)."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    old_denoised = None
    for i in tqdm(range(len(sigmas) - 1)):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
        h = t_next - t
        if old_denoised is None or sigmas[i + 1] == 0:
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised
        else:
            h_last = t - t_fn(sigmas[i - 1])
            r = h_last / h
            denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_d
        old_denoised = denoised
    return x


class SD3Inferencer:
    def print(self, txt):
        if self.verbose:
            print(txt)

    def load(self, model, vae, shift, verbose=False):
        self.verbose = verbose
        print("Loading tokenizers...")
        # NOTE: if you need a reference impl for a high performance CLIP tokenizer instead of just using the HF transformers one,
        # check https://github.com/Stability-AI/StableSwarmUI/blob/master/src/Utils/CliplikeTokenizer.cs
        # (T5 tokenizer is different though)
        self.tokenizer = SD3Tokenizer()
        print("Loading OpenAI CLIP L...")
        self.clip_l = ClipL()
        print("Loading OpenCLIP bigG...")
        self.clip_g = ClipG()
        print("Loading Google T5-v1-XXL...")
        self.t5xxl = T5XXL()
        print(f"Loading SD3 model {os.path.basename(model)}...")
        self.sd3 = SD3(model, shift, verbose)
        print("Loading VAE model...")
        self.vae = VAE(vae or model)
        print("Models loaded.")

    def get_empty_latent(self, width, height):
        self.print("Prep an empty latent...")
        return torch.ones(1, 16, height // 8, width // 8, device="cpu") * 0.0609

    def get_sigmas(self, sampling, steps):
        start = sampling.timestep(sampling.sigma_max)
        end = sampling.timestep(sampling.sigma_min)
        timesteps = torch.linspace(start, end, steps)
        sigs = []
        for x in range(len(timesteps)):
            ts = timesteps[x]
            sigs.append(sampling.sigma(ts))
        sigs += [0.0]
        return torch.FloatTensor(sigs)

    def get_cond(self, prompt):
        self.print("Encode prompt...")
        tokens = self.tokenizer.tokenize_with_weights(prompt)
        l_out, l_pooled = self.clip_l.model.encode_token_weights(tokens["l"])
        g_out, g_pooled = self.clip_g.model.encode_token_weights(tokens["g"])
        t5_out, t5_pooled = self.t5xxl.model.encode_token_weights(tokens["t5xxl"])
        lg_out = torch.cat([l_out, g_out], dim=-1)
        lg_out = torch.nn.functional.pad(lg_out, (0, 4096 - lg_out.shape[-1]))
        return torch.cat([lg_out, t5_out], dim=-2), torch.cat(
            (l_pooled, g_pooled), dim=-1
        )

    def max_denoise(self, sigmas):
        max_sigma = float(self.sd3.model.model_sampling.sigma_max)
        sigma = float(sigmas[0])
        return math.isclose(max_sigma, sigma, rel_tol=1e-05) or sigma > max_sigma

    def fix_cond(self, cond):
        cond, pooled = (cond[0].half().cuda(), cond[1].half().cuda())
        return {"c_crossattn": cond, "y": pooled}

    def do_sampling(
            self,
            noise,
            height,
            width,
            conditioning,
            neg_cond,
            steps,
            cfg_scale,
            sampler="dpmpp_2m",
            denoise=1.0,
            skip_layer_config={},
    ) -> torch.Tensor:
        self.print("Sampling...")
        latent = self.get_empty_latent(height, width)
        latent = latent.half().cuda()  # fp32->fp16
        self.sd3.model = self.sd3.model.cuda()
        sigmas = self.get_sigmas(self.sd3.model.model_sampling, steps).cuda()
        sigmas = sigmas[int(steps * (1 - denoise)):]
        conditioning = self.fix_cond(conditioning)  # conditioning fp32->fp16
        neg_cond = self.fix_cond(neg_cond)  # 空conditioning fp32->fp16
        extra_args = {"cond": conditioning, "uncond": neg_cond, "cond_scale": cfg_scale}
        noise_scaled = self.sd3.model.model_sampling.noise_scaling(
            sigmas[0], noise, latent, self.max_denoise(sigmas)
        )  # sigma * noise + (1.0 - sigma) * latent_image
        denoiser = (
            SkipLayerCFGDenoiser
            if skip_layer_config.get("scale", 0) > 0
            else CFGDenoiser
        )
        latent = sample_dpmpp_2m(
            denoiser(self.sd3.model, steps, skip_layer_config),
            noise_scaled,
            sigmas,
            extra_args=extra_args,
        )  # 和noise形状一样
        latent = SD3LatentFormat().process_out(latent)
        self.sd3.model = self.sd3.model.cpu()
        self.print("Sampling done")
        return latent

    def vae_encode(self, image) -> torch.Tensor:
        self.print("Encoding image to latent...")
        image = image.convert("RGB")
        image_np = np.array(image).astype(np.float32) / 255.0
        image_np = np.moveaxis(image_np, 2, 0)
        batch_images = np.expand_dims(image_np, axis=0).repeat(1, axis=0)
        image_torch = torch.from_numpy(batch_images)
        image_torch = 2.0 * image_torch - 1.0
        image_torch = image_torch.cuda()
        self.vae.model = self.vae.model.cuda()
        latent = self.vae.model.encode(image_torch).cpu()
        self.vae.model = self.vae.model.cpu()
        self.print("Encoded")
        return latent

    def process(self, x):
        x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)[0]
        x = np.moveaxis(x.cpu().numpy(), 0, 2)
        x = (x * 255.0).astype(np.uint8)  # 256,256,3
        return x

    def vae_decode(self, latent) -> torch.Tensor:
        self.print("Decoding latent to image...")
        latent = latent.cuda()
        self.vae.model = self.vae.model.cuda()
        image = self.vae.model.decode(latent)
        image = image.float()  # 1,3,256,256
        self.vae.model = self.vae.model.cpu()
        self.print("Decoded")
        return image
