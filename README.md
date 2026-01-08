# Diffusion-based Text-to-Video Generation (Hugging Face Diffusers)

A concise implementation of a diffusion-based text-to-video pipeline using Hugging Face Diffusers. The notebook was executed on GPU (Google Colab) and exports generated frames to an MP4 video. This README uses the second-last generated video bundled in this repository.

TL;DR: Text-to-video demo (80 frames, fp16, DPMSolverMultistepScheduler) — run the notebook in Colab for a reproducible demo.

## 🎬 Demo

<video width=720 controls>
  <source src="Canada_Downtown_Toronto.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

**Prompt used to generate this video:**
> A lakeshore view of Downtown Toronto Canada with city skyline reflected in Lake Ontario

## 🧠 Key technical details
- Model: damo-vilab/text-to-video-ms-1.7b (pretrained text-to-video diffusion)
- Framework: Hugging Face Diffusers
- Scheduler: DPMSolverMultistepScheduler
- Precision: fp16 (torch_dtype=torch.float16, variant="fp16")
- Device: GPU (pipe.to("cuda") if available)
- Notebook: `LLM_GEN_AI_text_to_video.ipynb`
- Output: MP4 (frames -> ffmpeg)

Typical example settings used for the demo:
- num_frames: 80 (≈10s at default FPS)
- num_inference_steps: 30
- height × width: 256 × 256
- guidance_scale: 7.5

## 📦 Dependencies
- Python 3.8+
- PyTorch with CUDA for GPU execution (if available)
- git+https://github.com/huggingface/diffusers
- transformers
- accelerate
- imageio[ffmpeg]
- ffmpeg (system package)

Quick pip commands (as used in the notebook):
```bash
pip install git+https://github.com/huggingface/diffusers
pip install transformers accelerate imageio[ffmpeg] -q
```
Ubuntu example for ffmpeg:
```bash
sudo apt update && sudo apt install -y ffmpeg
```

## ▶️ Quickstart (Colab)
Open and run the notebook in Colab: https://colab.research.google.com/github/dojedaro/Diffusion-based-text-to-video-generation-using-Hugging-Face-Diffusers/blob/main/LLM_GEN_AI_text_to_video.ipynb

Minimal steps to reproduce in the notebook:
1. Install dependencies.
2. Load the pipeline:
```python
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
```
3. Generate frames with your prompt:
```python
prompt = "A lakeshore view of Downtown Toronto Canada with city skyline reflected in Lake Ontario"
video_frames = pipe(prompt, num_inference_steps=30, num_frames=80, height=256, width=256, guidance_scale=7.5).frames
```
4. Save frames and encode to MP4 using imageio/ffmpeg (the notebook includes a working example).

## 🛠 Reproducibility tips & notes
- Reduce height/width or num_frames to fit limited VRAM.
- Lower num_inference_steps (20–30) for faster generation with slightly lower visual quality.
- Use fp16 and CUDA for significant speed and memory savings.
- If temporal jitter/artifacts appear, try increasing num_inference_steps or experimenting with a different scheduler or seed.

Recommended GIF preview command (to create a small preview):
```bash
ffmpeg -i assets/Canada_Downtown_Toronto.mp4 -vf "fps=12,scale=640:-1:flags=lanczos" -t 6 -y assets/preview.gif
```

## ⚠️ Ethics & usage
- Review the model card and license for damo-vilab/text-to-video-ms-1.7b on Hugging Face before commercial use.
- Be mindful of privacy, copyright, and potential misuse when generating or sharing synthetic media.
- Include clear disclaimers when presenting generated media in public-facing work.

## 📚 Credits
- Hugging Face Diffusers: https://github.com/huggingface/diffusers
- Model: damo-vilab/text-to-video-ms-1.7b (see the model card for citation details)

## 🔗 Contact
If you have questions or want to collaborate, open an issue or contact the repository owner.

----

*Committed by GitHub Copilot Chat Assistant.*
