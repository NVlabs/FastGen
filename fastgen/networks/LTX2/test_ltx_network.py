# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import numpy as np
from diffusers.pipelines.ltx2.export_utils import encode_video
from fastgen.networks.LTX2.network import LTX2


def test_ltx2_generation():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16  # LTX-2 is optimized for bfloat16

    # 1. Initialize the LTX2 model
    print("Initializing LTX-2 model...")
    model = LTX2(model_id="Lightricks/LTX-2", load_pretrained=True).to(device, dtype=dtype)
    model.init_preprocessors()
    model.eval()

    # 2. Prepare Prompts
    # prompt = "A high-performance sports car racing through a city street at night, neon lights reflecting off wet asphalt, motion blur streaking past the camera. The camera tracks low and close to the car as it accelerates aggressively, tires gripping the road, exhaust heat shimmering. Realistic lighting, cinematic depth of field, ultra-detailed textures, dynamic reflections, dramatic shadows, 4K realism, film-grade color grading."
    prompt = "A tight close-up shot of a musician's hands playing a grand piano. The audio is a fast-paced, uplifting classical piano sonata. The pressing of the black and white keys visually syncs with the rapid flurry of high-pitched musical notes. There is a slight echo, suggesting the piano is in a large, empty concert hall."
    prompt = "A street performer sitting on a brick stoop, strumming an acoustic guitar. The audio features a warm, indie-folk guitar chord progression. Over the guitar, a smooth, soulful human voice sings a slow, bluesy melody without words, just melodic humming and 'oohs'. The rhythmic strumming of the guitar perfectly matches the tempo of the vocal melody. Faint city traffic can be heard quietly in the deep background." 
    negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"

    print(f"Encoding prompt: {prompt}")
    # FIX 1: encode() returns (embeds, mask) tuple — unpack and move each tensor separately
    embeds, mask = model.text_encoder.encode(prompt, precision=dtype)
    condition = (embeds.to(device), mask.to(device))
    neg_embeds, neg_mask = model.text_encoder.encode(negative_prompt, precision=dtype)
    neg_condition = (neg_embeds.to(device), neg_mask.to(device))

    # 3. Define Video Parameters
    # LTX-2 video dimensions must be divisible by 32 spatially and (8n+1) temporally
    height, width = 480, 704
    num_frames = 81  # (8 * 10) + 1
    batch_size = 1

    # Calculate latent dimensions
    latent_f = (num_frames - 1) // model.vae_temporal_compression_ratio + 1
    latent_h = height // model.vae_spatial_compression_ratio
    latent_w = width // model.vae_spatial_compression_ratio

    # 4. Generate Initial Noise
    # FIX 2: model.vae IS AutoencoderKLLTX2Video directly — no .vae sub-attribute.
    #         Use transformer.config.in_channels as the pipeline does (line 989).
    latent_channels = model.transformer.config.in_channels  # 128

    # FIX 3: noise must be float32 — the pipeline creates latents in torch.float32
    #         (prepare_latents line 997) and keeps them float32 through the scheduler.
    #         Only cast to bfloat16 right before the transformer call.
    noise = torch.randn(
        batch_size, latent_channels, latent_f, latent_h, latent_w,
        device=device, dtype=torch.float32
    )

    # 5. Run Sampling (Inference)
    print("Starting sampling process...")
    with torch.no_grad():
        latents, audio_latents = model.sample(
            noise=noise,
            condition=condition,
            neg_condition=neg_condition,
            guidance_scale=4.0,
            num_steps=40,
            fps=24.0,
        )
    # latents:       [B, C, F, H, W] denormalised video latents (float32)
    # audio_latents: [B, C, L, M]    denormalised audio latents (float32)

    # 6. Decode Latents to Video
    print("Decoding latents to video...")
    with torch.no_grad():
        video_tensor = model.vae.decode(latents.to(model.vae.dtype))
        # model.vae.decode(latents.to(model.vae.dtype), return_dict=False)[0]
        # video_tensor: [B, C, F, H, W] in ~[-1, 1]

    # 7. Post-process and Save
    # Convert [B, C, F, H, W] -> [F, H, W, C] uint8
    video_np = (
        (video_tensor[0][0].cpu().float().permute(1, 2, 3, 0).numpy() + 1.0) * 127.5
    ).clip(0, 255).astype(np.uint8)


    print("Saving video to ltx2_test.mp4...")
    import imageio
    with imageio.get_writer("ltx2_test.mp4", fps=24, codec="libx264", quality=8) as writer:
        for frame in video_np:
            writer.append_data(frame)
    print("Done!")

if __name__ == "__main__":
    test_ltx2_generation()