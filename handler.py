import os, base64, tempfile
import runpod
import torch
import imageio
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

# Pick a smaller model first; 5B needs more VRAM.
# You can switch to "THUDM/CogVideoX-5b" later if your GPU can handle it.
MODEL_ID = os.environ.get("MODEL_ID", "THUDM/CogVideoX-2b")

pipe = None

def get_pipe():
    global pipe
    if pipe is None:
        dtype = torch.float16
        pipe = CogVideoXPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype)
        pipe.to("cuda")
        # Optional speed/memory tweak:
        # pipe.enable_model_cpu_offload()  # if VRAM is tight (slower)
    return pipe

def handler(job):
    job_input = job["input"]

    prompt = job_input.get("prompt", "A cinematic shot of a landscape, smooth camera motion")
    negative_prompt = job_input.get("negative_prompt", None)

    height = int(job_input.get("height", 480))
    width = int(job_input.get("width", 720))
    num_frames = int(job_input.get("num_frames", 49))  # keep small for MVP
    steps = int(job_input.get("steps", 30))
    guidance_scale = float(job_input.get("guidance_scale", 6.0))
    fps = int(job_input.get("fps", 8))
    seed = job_input.get("seed", None)

    generator = None
    if seed is not None:
        generator = torch.Generator(device="cuda").manual_seed(int(seed))

    pipe = get_pipe()

    with torch.inference_mode():
        out = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )

    # Diffusers returns frames; export to mp4
    with tempfile.TemporaryDirectory() as td:
        mp4_path = os.path.join(td, "out.mp4")
        export_to_video(out.frames[0], mp4_path, fps=fps)

        with open(mp4_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")

    return {"video_base64": b64, "mime": "video/mp4"}

runpod.serverless.start({"handler": handler})
