# Free AI video upscaling on a MacBook Air M2: a practical guide

**The best free approach for upscaling short live-action clips on an M2 MacBook Air is the `realesrgan-ncnn-vulkan` CLI binary paired with ffmpeg**, using the `realesrgan-x4plus` model. It runs natively on Apple Silicon via Vulkan/MoltenVK, processes a 15-second 30fps clip in roughly 15–37 minutes, and produces genuinely impressive detail recovery on faces and textures. For users who want speed over maximum quality, **fx-upscale** (a MetalFX-based tool installable via Homebrew) processes video directly without frame extraction and finishes in seconds — though its upscaling quality sits below dedicated AI models. A CoreML/Neural Engine pathway exists that combines Real-ESRGAN quality with near-real-time speed, but requires more setup. Here's everything you need to know to pick the right tool and build a working pipeline.

## Six tools that actually work on Apple Silicon

After testing compatibility across every major free upscaling project, six tools genuinely run on an M2 MacBook Air with 16GB RAM. Several popular options — Video2X (no native macOS support), SUPIR (requires NVIDIA CUDA and 12GB+ VRAM), and ffmpeg's built-in DNN filters (no Apple hardware acceleration, primitive models only) — are effectively unusable for this workflow.

**realesrgan-ncnn-vulkan** is the foundational CLI tool that most GUI apps wrap. The macOS binary (v0.2.5.0, April 2022 — still the latest release) runs on Apple Silicon via Vulkan translated through MoltenVK to Metal. It ships with four models and processes entire directories of frames in a single command. Download it directly from the [Real-ESRGAN GitHub releases](https://github.com/xinntao/Real-ESRGAN/releases), `chmod u+x` the binary, clear Gatekeeper quarantine with `xattr -cr realesrgan-ncnn-vulkan`, and you're running.

**fx-upscale** deserves special attention as a Mac-native option. Built on Apple's MetalFX framework (Apple's answer to NVIDIA DLSS), it installs with a single `brew install finnvoor/tools/fx-upscale` and processes video files directly — no frame extraction needed. One user upscaled a 38-minute 4K video to 8K in about an hour on an M1 Pro. The quality is good but not AI-model-level; MetalFX analyzes edges and textures rather than hallucinating new detail the way Real-ESRGAN does.

**Upscayl** is a polished GUI app (free from GitHub, $12 on the Mac App Store) that wraps realesrgan-ncnn-vulkan with drag-and-drop simplicity. It handles images only — not video directly — but its batch processing mode can upscale a folder of extracted frames. It includes extra models like **Remacri**, **UltraSharp**, and a newer **High Fidelity** model beyond the standard Real-ESRGAN set. A known memory leak in v2.11.5 can crash the app after processing 1,000–5,000 images, so splitting frame batches is wise.

**Upscale-Enhance** (open-source, MIT license) converts Real-ESRGAN to CoreML format and runs inference on the **Apple Neural Engine**, claiming up to **78× speedup** over CPU-based PyTorch. It supports both image and video, runs on M1/M2 without an NVIDIA GPU, and is available on the Mac App Store. This is the fastest AI-quality option on Apple Silicon.

**FreeScaler CoreML** takes a similar approach — a free macOS-native app (also on the App Store) that uses CoreML for Neural Engine acceleration with 75% lower power consumption than GPU processing. It has an experimental movie upscaling feature for `.mov` files and supports custom CoreML models.

**REAL Video Enhancer** is a true video upscaler (not image-only) with a native macOS GUI for both Intel and Apple Silicon (requires macOS 14+). It uses NCNN/Vulkan and ships curated models including **4x-SPANkendata** and **Nomos8k-SPAN** optimized for general video content, plus RIFE for frame interpolation.

## Choosing the right model for live-action faces

The model choice matters enormously. The ncnn-vulkan binary ships four models, but only one is designed for real-world footage with people:

| Model | Use case | Quality for faces | Speed |
|-------|----------|-------------------|-------|
| **realesrgan-x4plus** | General photos, live-action | ★★★★★ Best | Slowest (~2–5 sec/frame) |
| realesrnet-x4plus | General (MSE-trained) | ★★☆☆☆ Over-smooth, waxy skin | Slow |
| realesr-animevideov3 | Anime video | ★★☆☆☆ Makes faces look artificial | Fast |
| realesrgan-x4plus-anime | Anime images | ★☆☆☆☆ Not for real people | Medium |

**Use `realesrgan-x4plus` for live-action content with people talking or dancing.** It's the flagship model trained with GAN loss for perceptual quality — it reconstructs skin textures, hair detail, and clothing fabric far better than any alternative in this binary. The MSE-trained `realesrnet-x4plus` produces plastic-looking faces and should be avoided.

For face-heavy content, the Python inference script (not the ncnn binary) supports a `--face_enhance` flag that integrates **GFPGAN** face restoration. This dramatically improves facial detail but requires a full PyTorch setup. **CodeFormer** is a more advanced alternative with a tunable fidelity parameter, though it's not integrated into the Real-ESRGAN pipeline directly. Neither face enhancement option is available through the ncnn-vulkan binary — only through the Python path.

A useful quality trick: **upscale at 4× then downscale to your target resolution with Lanczos**. The 4× model produces richer detail than a 2× model, and the subsequent downscaling adds natural anti-aliasing. For 720p→1080p (a ~1.5× increase), this 4×-then-shrink approach yields visibly better results than a direct 2× upscale.

## Real performance numbers on M2 MacBook Air

The M2 MacBook Air's fanless design is the elephant in the room. Under sustained GPU load, **thermal throttling kicks in after 5–6 minutes**, dropping performance roughly 13% and settling near M1-level speeds. For a 15-second clip this is manageable; for longer content it becomes a real constraint.

| Approach | Time per frame (720p, 4×) | Time for 450 frames (~15 sec video) |
|----------|---------------------------|--------------------------------------|
| ncnn-vulkan (GPU via MoltenVK) | **2–5 seconds** | **15–37 minutes** |
| PyTorch MPS (Metal GPU) | 5–15 seconds | 37 min – 1.9 hours |
| CoreML / Neural Engine | **0.5–1.5 seconds** | **4–11 minutes** |
| PyTorch CPU-only | 15–45 seconds | 1.9–5.6 hours |
| fx-upscale (MetalFX, non-AI) | Near real-time | **Under 1 minute** |

**Memory is not a bottleneck.** All tile-based tools keep usage well under 16GB. The ncnn-vulkan binary auto-selects tile size; start with the default (auto/`-t 0`) and drop to `-t 200` only if you see crashes. For PyTorch MPS, set `tile=256` and use `torch.mps.empty_cache()` between frames. Disk space matters more: 450 upscaled PNG frames at 4× from 720p will consume roughly **6–7GB** of temporary storage.

The CoreML/Neural Engine path is the performance sweet spot — AI-quality upscaling at 2–5 fps with minimal heat generation because inference runs on the dedicated Neural Engine rather than the GPU. The Upscale-Enhance developer reports **2–5 fps on M2 Pro** with CoreML, and the M2 has the same 15.8 TOPS Neural Engine.

## Complete end-to-end pipeline for Instagram Reels

Here is the tested workflow from source video to Instagram-ready 1080×1920 MP4, using the ncnn-vulkan binary. Every command is copy-paste ready.

**Install dependencies:**
```bash
brew install ffmpeg
curl -L -o realesrgan.zip \
  https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-macos.zip
unzip realesrgan.zip -d realesrgan && cd realesrgan
chmod u+x realesrgan-ncnn-vulkan
xattr -cr realesrgan-ncnn-vulkan
```

**Step 1 — Probe source video and extract frames:**
```bash
# Check source specs
ffprobe -v error -select_streams v:0 \
  -show_entries stream=width,height,r_frame_rate -of csv=p=0 input.mp4

# Extract frames as lossless PNGs
mkdir -p tmp_frames out_frames
ffmpeg -i input.mp4 -qscale:v 1 -qmin 1 -qmax 1 -vsync passthrough tmp_frames/frame_%08d.png
```

**Step 2 — Upscale with Real-ESRGAN:**
```bash
./realesrgan-ncnn-vulkan \
  -i tmp_frames \
  -o out_frames \
  -n realesrgan-x4plus \
  -s 4 \
  -f png
```

**Step 3 — Reassemble and encode for Instagram Reels (1080×1920, H.264):**

If your source is already vertical (9:16), the upscaled frames just need scaling down to 1080×1920:
```bash
ffmpeg -framerate 30 -i out_frames/frame_%08d.png \
  -i input.mp4 \
  -map 0:v:0 -map 1:a:0? \
  -vf "scale=1080:1920:flags=lanczos" \
  -c:v libx264 -profile:v main -level:v 4.0 \
  -pix_fmt yuv420p \
  -crf 18 -maxrate 12M -bufsize 20M \
  -r 30 -preset slow \
  -c:a aac -b:a 192k -ar 48000 \
  -movflags +faststart \
  output_reels.mp4
```

If your source is horizontal (16:9) and needs converting to vertical 9:16, choose one of these `-vf` chains instead:

```bash
# Center-crop to 9:16 (loses sides, keeps sharpness)
-vf "scale=-2:1920:flags=lanczos,crop=1080:1920"

# Blurred-background fill (popular Reels style, keeps full frame)
-vf "[0:v]split[fg][bg]; \
  [bg]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=20[bg]; \
  [fg]scale=-2:1920:force_original_aspect_ratio=decrease[fg]; \
  [bg][fg]overlay=(W-w)/2:(H-h)/2"
```

**Step 4 — Clean up:**
```bash
rm -rf tmp_frames out_frames
```

For **hardware-accelerated encoding** (4× faster, negligible quality difference since Instagram re-compresses anyway), replace `-c:v libx264 ... -preset slow` with:
```bash
-c:v h264_videotoolbox -b:v 12M
```

## The PyTorch MPS and CoreML alternative paths

For maximum quality on face-heavy content, the Python/PyTorch path unlocks GFPGAN face enhancement. PyTorch's MPS backend on Apple Silicon delivers **8–10× speedup over CPU** but remains slower than ncnn-vulkan. Setup:

```bash
brew install miniconda
conda create -n esrgan python=3.12 && conda activate esrgan
pip install torch torchvision realesrgan basicsr gfpgan

# Verify MPS works
python -c "import torch; print(torch.backends.mps.is_available())"
```

The official Real-ESRGAN repo has **not merged MPS support** (GitHub Issue #902 is still open). You must manually patch the device selection in `inference_realesrgan.py` to use `torch.device("mps")` and always pass `--fp32` since MPS doesn't support fp16 for this model. With that patch:

```bash
python inference_realesrgan.py \
  -i tmp_frames -o out_frames \
  -n RealESRGAN_x4plus -s 4 --outscale 2 \
  --face_enhance --fp32
```

The **CoreML/Neural Engine** path is the most compelling for anyone willing to invest 30 minutes in setup. Convert the PyTorch model to CoreML using `coremltools` and `spandrel`, then run inference through **Upscale-Enhance** or **FreeScaler 2**. The Neural Engine runs at 15.8 TOPS on M2 with minimal heat generation — critical for the fanless MacBook Air. A March 2025 guide by Ron Regev documents this conversion process in detail, specifically recommending the `realesr-general-x4v3` compact model for the CoreML path.

## Avoiding the three biggest pitfalls

**Temporal flickering** is the most visible artifact in frame-by-frame AI upscaling. Each frame is processed independently, so subtle differences in how the model reconstructs textures create frame-to-frame inconsistency. There's no perfect solution in the free toolchain — professional tools like Topaz use temporal models to address this. Mitigation strategies: use consistent tile sizes across all frames, consider the compact `realesr-animevideov3` model (which despite its name has better temporal consistency due to its lighter architecture), and apply ffmpeg's `minterpolate` or `deflicker` filters as a post-processing step.

**Thermal throttling** will slow your M2 Air after 5–6 minutes of sustained upscaling. Elevate the laptop for airflow, process in a cool room, and for longer clips consider the CoreML/Neural Engine path (which offloads work to the dedicated ML hardware and generates significantly less heat than GPU processing). The ncnn-vulkan binary's `-j 1:1:1` flag reduces thread count and heat at the cost of some speed.

**Disk space** catches people off guard. A 15-second 30fps video produces 450 frames. At 4× upscale from 720p, each output PNG is 10–30MB, totaling **4.5–13.5GB** for the upscaled frames alone plus another 1–2GB for the input frames. Ensure at least 20GB of free space before starting.

## Conclusion

For a MacBook Air M2 owner comfortable with Terminal, the **ncnn-vulkan binary + ffmpeg pipeline** is the battle-tested default: it works today with a 5-minute setup, produces high-quality results on live-action faces using `realesrgan-x4plus`, and processes a 15-second Reel in under 40 minutes. The newer **CoreML/Neural Engine tools** (Upscale-Enhance, FreeScaler) represent the future on Apple Silicon — same Real-ESRGAN quality at 5–10× the speed with less thermal stress — but require either a model conversion step or trusting a third-party app. **fx-upscale** fills a different niche entirely: when you need "good enough" upscaling in seconds rather than minutes, its MetalFX-based approach is unbeatable for speed. The one gap no free tool fully solves is temporal consistency — for now, that remains the price of frame-by-frame processing, and the primary advantage paid tools still hold.
