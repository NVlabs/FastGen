import torch
import numpy as np
import imageio

from pipeline_ltx2 import LTX2Pipeline
from transformer_ltx2 import LTX2VideoTransformer3DModel

device = "cuda:0"
width = 768
height = 512

# 1. Load the full pipeline (vae, scheduler, text_encoder, etc.)
pipe = LTX2Pipeline.from_pretrained("Lightricks/LTX-2", torch_dtype=torch.bfloat16)

# 2. Build a video-only transformer — no audio weights allocated
config = pipe.transformer.config
transformer = LTX2VideoTransformer3DModel.from_config(config, audio_enabled=False)
transformer = transformer.to(torch.bfloat16)

# 3. Copy video weights from the pretrained AV transformer, skip audio keys
state_dict = pipe.transformer.state_dict()
missing, unexpected = transformer.load_state_dict(state_dict, strict=False)
assert len(missing) == 0, f"Missing video keys: {missing}"
print(f"Skipped {len(unexpected)} audio keys from checkpoint")

# 4. Swap in the video-only transformer and free the original AV one
del pipe.transformer
pipe.transformer = transformer

pipe.to(device)  # fully on GPU — fastest, needs ~24GB+ VRAM
# pipe.enable_model_cpu_offload(device=device)  # offloads whole models between steps, needs ~12GB VRAM

prompt = "A beautiful sunset over the ocean"
negative_prompt = "shaky, glitchy, low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, fused fingers, bad anatomy, weird hand, ugly, transition, static."

frame_rate = 24.0
video, audio = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=width,
    height=height,
    num_frames=121,
    frame_rate=frame_rate,
    num_inference_steps=40,
    guidance_scale=4.0,
    output_type="np",
    return_dict=False,
    generate_audio=False,  # audio is None, no audio ops run
)
print(audio)

# video[0] is (num_frames, height, width, 3) float in [0, 1]
frames = (video[0] * 255).clip(0, 255).astype(np.uint8)

output_path = "ltx2_video_only.mp4"
with imageio.get_writer(output_path, fps=frame_rate, codec="libx264", quality=8) as writer:
    for frame in frames:
        writer.append_data(frame)

print(f"Saved to {output_path}")