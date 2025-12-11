"""
THis is the main file for the gradio web demo. It uses the CogVideoX-5B model to generate videos gradio web demo.
set environment variable OPENAI_API_KEY to use the OpenAI API to enhance the prompt.

Usage:
    OpenAI_API_KEY=your_openai_api_key OPENAI_BASE_URL=https://api.openai.com/v1 python inference/gradio_web_demo.py
"""

print("=" * 60)
print("🚀 COGVIDEO APP STARTING...")
print("=" * 60)

import math
import os
import random
import sys
import threading
import time
from pathlib import Path

import cv2
import tempfile
import imageio_ffmpeg
import gradio as gr
import torch
from PIL import Image
from diffusers import (
    CogVideoXPipeline,
    CogVideoXDPMScheduler,
    CogVideoXVideoToVideoPipeline,
    CogVideoXImageToVideoPipeline,
    CogVideoXTransformer3DModel,
)
from diffusers.utils import load_video, load_image
from datetime import datetime, timedelta

from diffusers.image_processor import VaeImageProcessor
from openai import OpenAI
from moviepy import VideoFileClip
import utils
from rife_model import load_rife_model, rife_inference_with_latents
from huggingface_hub import hf_hub_download, snapshot_download

# Add parent directory to path to import fps_utils
print("1. Configuring sys.path...")
sys.path.insert(0, str(Path(__file__).parent.parent))
print("2. Importing fps_utils...")
from fps_utils import get_correct_fps
print("3. Importing vram_utils...")
from vram_utils import get_recommended_offload_strategy, apply_offload_strategy, configure_vae_tiling
print("4. Importing memory_manager...")
from memory_manager import get_memory_manager
print("5. Checking CUDA availability...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"   Device: {device}")

# Initialize Memory Manager with aggressive cleanup for production
print("6. Initializing Memory Manager...")
memory_manager = get_memory_manager(aggressive_cleanup=True)
print("   ✅ Memory Manager initialized")

MODEL_T2V = "THUDM/CogVideoX-5b"
MODEL_I2V = "THUDM/CogVideoX-5b-I2V"

# Models for upscaling and interpolation will be downloaded on-demand when first used
print("✅ CogVideo app starting...")


# Pipeline loader functions for Memory Manager
def _load_t2v_pipeline():
    """Load Text-to-Video pipeline."""
    print("📥 Loading T2V pipeline...")
    model_path = MODEL_T2V
    offload_strategy = get_recommended_offload_strategy(model_path)
    
    pipe = CogVideoXPipeline.from_pretrained(MODEL_T2V, torch_dtype=torch.bfloat16)
    apply_offload_strategy(pipe, offload_strategy)
    pipe.scheduler = CogVideoXDPMScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing"
    )
    configure_vae_tiling(pipe.vae)
    print(f"✅ T2V pipeline loaded with {offload_strategy} offload")
    return pipe

def _load_i2v_pipeline():
    """Load Image-to-Video pipeline."""
    print("📥 Loading I2V pipeline...")
    model_path = MODEL_I2V
    offload_strategy = get_recommended_offload_strategy(model_path)
    
    pipe = CogVideoXImageToVideoPipeline.from_pretrained(
        MODEL_I2V, torch_dtype=torch.bfloat16
    )
    apply_offload_strategy(pipe, offload_strategy)
    pipe.scheduler = CogVideoXDPMScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing"
    )
    configure_vae_tiling(pipe.vae)
    print(f"✅ I2V pipeline loaded with {offload_strategy} offload")
    return pipe

def _load_v2v_pipeline():
    """Load Video-to-Video pipeline."""
    print("📥 Loading V2V pipeline...")
    model_path = MODEL_T2V
    offload_strategy = get_recommended_offload_strategy(model_path)
    
    pipe = CogVideoXVideoToVideoPipeline.from_pretrained(
        MODEL_T2V, torch_dtype=torch.bfloat16
    )
    apply_offload_strategy(pipe, offload_strategy)
    pipe.scheduler = CogVideoXDPMScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing"
    )
    configure_vae_tiling(pipe.vae)
    print(f"✅ V2V pipeline loaded with {offload_strategy} offload")
    return pipe

# Register ALL pipelines with Memory Manager
print("🔧 Registering pipelines with Memory Manager...")
memory_manager.register_model_loader("pipeline_t2v", _load_t2v_pipeline)
memory_manager.register_model_loader("pipeline_i2v", _load_i2v_pipeline)
memory_manager.register_model_loader("pipeline_v2v", _load_v2v_pipeline)
print("✅ Pipelines registered")


# Register model loaders with Memory Manager
def _load_upscale_model():
    """Model loader function for upscaling."""
    print("Loading upscaling model (Real-ESRGAN)...")
    # Download if not exists
    if not os.path.exists("model_real_esran/RealESRGAN_x4.pth"):
        print("  Downloading Real-ESRGAN model...")
        hf_hub_download(
            repo_id="ai-forever/Real-ESRGAN", filename="RealESRGAN_x4.pth", local_dir="model_real_esran"
        )
    return utils.load_sd_upscale("model_real_esran/RealESRGAN_x4.pth", device)

def _load_frame_interpolation_model():
    """Model loader function for frame interpolation."""
    print("Loading frame interpolation model (RIFE)...")
    # Download if not exists
    if not os.path.exists("model_rife"):
        print("  Downloading RIFE model...")
        snapshot_download(repo_id="AlexWortega/RIFE", local_dir="model_rife")
    return load_rife_model("model_rife")

# Register loaders with Memory Manager
memory_manager.register_model_loader("upscale", _load_upscale_model)
memory_manager.register_model_loader("frame_interpolation", _load_frame_interpolation_model)

def cleanup_models():
    """Free memory from all models and clear CUDA cache."""
    try:
        memory_manager.unload_all_models()
        
        import gc
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
        print("✅ Models cleaned up successfully.")
        
    except Exception as e:
        print(f"Error during cleanup: {str(e)}")

def cleanup_on_error():
    """Emergency cleanup function to be called on errors."""
    print("\n⚠️ ERROR DETECTED - Performing emergency memory cleanup...")
    
    try:
        memory_manager.force_cleanup()
        
        import gc
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            
        print("✅ Emergency cleanup completed.\n")
        
    except Exception as e:
        print(f"❌ Error during emergency cleanup: {str(e)}")

# Create output directories
os.makedirs("./output", exist_ok=True)
os.makedirs("./gradio_tmp", exist_ok=True)

sys_prompt = """You are part of a team of bots that creates videos. You work with an assistant bot that will draw anything you say in square brackets.

For example , outputting " a beautiful morning in the woods with the sun peaking through the trees " will trigger your partner bot to output an video of a forest morning , as described. You will be prompted by people looking to create detailed , amazing videos. The way to accomplish this is to take their short prompts and make them extremely detailed and descriptive.
There are a few rules to follow:

You will only ever output a single video description per user request.

When modifications are requested , you should not simply make the description longer . You should refactor the entire description to integrate the suggestions.
Other times the user will not want modifications , but instead want a new image . In this case , you should ignore your previous conversation with the user.

Video descriptions must have the same num of words as examples below. Extra words will be ignored.
"""


def resize_if_unfit(input_video, progress=gr.Progress(track_tqdm=True)):
    width, height = get_video_dimensions(input_video)

    if width == 720 and height == 480:
        processed_video = input_video
    else:
        processed_video = center_crop_resize(input_video)
    return processed_video


def get_video_dimensions(input_video_path):
    reader = imageio_ffmpeg.read_frames(input_video_path)
    metadata = next(reader)
    return metadata["size"]


def center_crop_resize(input_video_path, target_width=720, target_height=480):
    cap = cv2.VideoCapture(input_video_path)

    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    width_factor = target_width / orig_width
    height_factor = target_height / orig_height
    resize_factor = max(width_factor, height_factor)

    inter_width = int(orig_width * resize_factor)
    inter_height = int(orig_height * resize_factor)

    target_fps = 8
    ideal_skip = max(0, math.ceil(orig_fps / target_fps) - 1)
    skip = min(5, ideal_skip)  # Cap at 5

    while (total_frames / (skip + 1)) < 49 and skip > 0:
        skip -= 1

    processed_frames = []
    frame_count = 0
    total_read = 0

    while frame_count < 49 and total_read < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if total_read % (skip + 1) == 0:
            resized = cv2.resize(frame, (inter_width, inter_height), interpolation=cv2.INTER_AREA)

            start_x = (inter_width - target_width) // 2
            start_y = (inter_height - target_height) // 2
            cropped = resized[start_y : start_y + target_height, start_x : start_x + target_width]

            processed_frames.append(cropped)
            frame_count += 1

        total_read += 1

    cap.release()

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
        temp_video_path = temp_file.name
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(temp_video_path, fourcc, target_fps, (target_width, target_height))

        for frame in processed_frames:
            out.write(frame)

        out.release()

    return temp_video_path


def convert_prompt(prompt: str, retry_times: int = 3) -> str:
    if not os.environ.get("OPENAI_API_KEY"):
        return prompt
    client = OpenAI()
    text = prompt.strip()

    for i in range(retry_times):
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": sys_prompt},
                {
                    "role": "user",
                    "content": 'Create an imaginative video descriptive caption or modify an earlier caption for the user input : "a girl is on the beach"',
                },
                {
                    "role": "assistant",
                    "content": "A radiant woman stands on a deserted beach, arms outstretched, wearing a beige trench coat, white blouse, light blue jeans, and chic boots, against a backdrop of soft sky and sea. Moments later, she is seen mid-twirl, arms exuberant, with the lighting suggesting dawn or dusk. Then, she runs along the beach, her attire complemented by an off-white scarf and black ankle boots, the tranquil sea behind her. Finally, she holds a paper airplane, her pose reflecting joy and freedom, with the ocean's gentle waves and the sky's soft pastel hues enhancing the serene ambiance.",
                },
                {
                    "role": "user",
                    "content": 'Create an imaginative video descriptive caption or modify an earlier caption for the user input : "A man jogging on a football field"',
                },
                {
                    "role": "assistant",
                    "content": "A determined man in athletic attire, including a blue long-sleeve shirt, black shorts, and blue socks, jogs around a snow-covered soccer field, showcasing his solitary exercise in a quiet, overcast setting. His long dreadlocks, focused expression, and the serene winter backdrop highlight his dedication to fitness. As he moves, his attire, consisting of a blue sports sweatshirt, black athletic pants, gloves, and sneakers, grips the snowy ground. He is seen running past a chain-link fence enclosing the playground area, with a basketball hoop and children's slide, suggesting a moment of solitary exercise amidst the empty field.",
                },
                {
                    "role": "user",
                    "content": 'Create an imaginative video descriptive caption or modify an earlier caption for the user input : " A woman is dancing, HD footage, close-up"',
                },
                {
                    "role": "assistant",
                    "content": "A young woman with her hair in an updo and wearing a teal hoodie stands against a light backdrop, initially looking over her shoulder with a contemplative expression. She then confidently makes a subtle dance move, suggesting rhythm and movement. Next, she appears poised and focused, looking directly at the camera. Her expression shifts to one of introspection as she gazes downward slightly. Finally, she dances with confidence, her left hand over her heart, symbolizing a poignant moment, all while dressed in the same teal hoodie against a plain, light-colored background.",
                },
                {
                    "role": "user",
                    "content": f'Create an imaginative video descriptive caption or modify an earlier caption in ENGLISH for the user input: "{text}"',
                },
            ],
            model="glm-4-plus",
            temperature=0.01,
            top_p=0.7,
            stream=False,
            max_tokens=200,
        )
        if response.choices:
            return response.choices[0].message.content
    return prompt


def infer(
    prompt: str,
    image_input: str,
    video_input: str,
    video_strenght: float,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int = -1,
    progress=gr.Progress(track_tqdm=True),
):
    """
    Run inference with appropriate pipeline based on input type.
    
    Returns:
        tuple: (video_pt, seed, model_path) where model_path is the model used for FPS detection
    """
    try:
        if seed == -1:
            seed = random.randint(0, 2**8 - 1)

        if video_input is not None:
            print(f"🎬 Running V2V inference with seed {seed}")
            with memory_manager.load_model("pipeline_v2v") as pipe_v2v:
                video = load_video(video_input)[:49]  # Limit to 49 frames
                video_pt = pipe_v2v(
                    video=video,
                    prompt=prompt,
                    num_inference_steps=num_inference_steps,
                    num_videos_per_prompt=1,
                    strength=video_strenght,
                    use_dynamic_cfg=True,
                    output_type="pt",
                    guidance_scale=guidance_scale,
                    generator=torch.Generator(device="cpu").manual_seed(seed),
                ).frames
            # Pipeline automaticamente descarregado aqui
            print("✅ V2V pipeline unloaded")
            return (video_pt, seed, MODEL_T2V)
            
        elif image_input is not None:
            print(f"🎬 Running I2V inference with seed {seed}")
            with memory_manager.load_model("pipeline_i2v") as pipe_i2v:
                image_input = Image.fromarray(image_input).resize(size=(720, 480))  # Convert to PIL
                image = load_image(image_input)
                video_pt = pipe_i2v(
                    image=image,
                    prompt=prompt,
                    num_inference_steps=num_inference_steps,
                    num_videos_per_prompt=1,
                    use_dynamic_cfg=True,
                    output_type="pt",
                    guidance_scale=guidance_scale,
                    generator=torch.Generator(device="cpu").manual_seed(seed),
                ).frames
            # Pipeline automaticamente descarregado aqui
            print("✅ I2V pipeline unloaded")
            return (video_pt, seed, MODEL_I2V)
            
        else:
            print(f"🎬 Running T2V inference with seed {seed}")
            with memory_manager.load_model("pipeline_t2v") as pipe_t2v:
                video_pt = pipe_t2v(
                    prompt=prompt,
                    num_videos_per_prompt=1,
                    num_inference_steps=num_inference_steps,
                    num_frames=49,
                    use_dynamic_cfg=True,
                    output_type="pt",
                    guidance_scale=guidance_scale,
                    generator=torch.Generator(device="cpu").manual_seed(seed),
                ).frames
            # Pipeline automaticamente descarregado aqui
            print("✅ T2V pipeline unloaded")
            return (video_pt, seed, MODEL_T2V)
            
    except torch.cuda.OutOfMemoryError as e:
        print(f"\n❌ GPU Out of Memory during inference: {str(e)}")
        raise
    except Exception as e:
        print(f"\n❌ Error during inference: {str(e)}")
        print(f"Exception type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        raise


def convert_to_gif(video_path):
    clip = VideoFileClip(video_path)
    clip = clip.with_fps(8)
    clip = clip.resized(height=240)
    gif_path = video_path.replace(".mp4", ".gif")
    clip.write_gif(gif_path, fps=8)
    return gif_path


def delete_old_files():
    """Delete old files and perform memory cleanup periodically."""
    while True:
        now = datetime.now()
        cutoff = now - timedelta(minutes=10)
        directories = ["./output", "./gradio_tmp"]

        for directory in directories:
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                if os.path.isfile(file_path):
                    file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                    if file_mtime < cutoff:
                        os.remove(file_path)
        
        # Cleanup de memória a cada 10 minutos
        print("🧹 Periodic memory cleanup...")
        memory_manager.auto_cleanup_if_needed(threshold_percent=70.0)
        
        time.sleep(600)  # 10 minutos


threading.Thread(target=delete_old_files, daemon=True).start()
print("✅ Background cleanup thread started")
examples_videos = [
    ["example_videos/horse.mp4"],
    ["example_videos/kitten.mp4"],
    ["example_videos/train_running.mp4"],
]
examples_images = [
    ["example_images/beach.png"],
    ["example_images/street.png"],
    ["example_images/camping.png"],
]

with gr.Blocks() as demo:
    gr.Markdown("""
           <div style="text-align: center; font-size: 32px; font-weight: bold; margin-bottom: 20px;">
               CogVideoX-5B Huggingface Space🤗
           </div>
           <div style="text-align: center;">
               <a href="https://huggingface.co/THUDM/CogVideoX-5B">🤗 5B(T2V) Model Hub</a> |
               <a href="https://huggingface.co/THUDM/CogVideoX-5B-I2V">🤗 5B(I2V) Model Hub</a> |
               <a href="https://github.com/THUDM/CogVideo">🌐 Github</a> |
               <a href="https://arxiv.org/pdf/2408.06072">📜 arxiv </a>
           </div>
           <div style="text-align: center;display: flex;justify-content: center;align-items: center;margin-top: 1em;margin-bottom: .5em;">
              <span>If the Space is too busy, duplicate it to use privately</span>
              <a href="https://huggingface.co/spaces/THUDM/CogVideoX-5B-Space?duplicate=true"><img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/duplicate-this-space-lg.svg" width="160" style="
                margin-left: .75em;
            "></a>
           </div>
           <div style="text-align: center; font-size: 15px; font-weight: bold; color: red; margin-bottom: 20px;">
            ⚠️ This demo is for academic research and experimental use only.
            </div>
           """)
    with gr.Row():
        with gr.Column():
            with gr.Accordion(
                "I2V: Image Input (cannot be used simultaneously with video input)", open=False
            ):
                image_input = gr.Image(label="Input Image (will be cropped to 720 * 480)")
                examples_component_images = gr.Examples(
                    examples_images, inputs=[image_input], cache_examples=False
                )
            with gr.Accordion(
                "V2V: Video Input (cannot be used simultaneously with image input)", open=False
            ):
                video_input = gr.Video(
                    label="Input Video (will be cropped to 49 frames, 6 seconds at 8fps)"
                )
                strength = gr.Slider(0.1, 1.0, value=0.8, step=0.01, label="Strength")
                examples_component_videos = gr.Examples(
                    examples_videos, inputs=[video_input], cache_examples=False
                )
            prompt = gr.Textbox(
                label="Prompt (Less than 200 Words)", placeholder="Enter your prompt here", lines=5
            )

            with gr.Row():
                gr.Markdown(
                    "✨Upon pressing the enhanced prompt button, we will use [GLM-4 Model](https://github.com/THUDM/GLM-4) to polish the prompt and overwrite the original one."
                )
                enhance_button = gr.Button("✨ Enhance Prompt(Optional)")
            with gr.Group():
                with gr.Column():
                    with gr.Row():
                        seed_param = gr.Number(
                            label="Inference Seed (Enter a positive number, -1 for random)",
                            value=-1,
                        )
                    with gr.Row():
                        enable_scale = gr.Checkbox(
                            label="Super-Resolution (720 × 480 -> 2880 × 1920)", value=False
                        )
                        enable_rife = gr.Checkbox(
                            label="Frame Interpolation (8fps -> 16fps)", value=False
                        )
                    gr.Markdown(
                        "✨In this demo, we use [RIFE](https://github.com/hzwer/ECCV2022-RIFE) for frame interpolation and [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) for upscaling(Super-Resolution).<br>&nbsp;&nbsp;&nbsp;&nbsp;The entire process is based on open-source solutions."
                    )

            generate_button = gr.Button("🎬 Generate Video")

        with gr.Column():
            video_output = gr.Video(label="CogVideoX Generate Video", width=720, height=480)
            with gr.Row():
                download_video_button = gr.File(label="📥 Download Video", visible=False)
                download_gif_button = gr.File(label="📥 Download GIF", visible=False)
                seed_text = gr.Number(label="Seed Used for Video Generation", visible=False)

    gr.Markdown("""
    <table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
        <div style="text-align: center; font-size: 32px; font-weight: bold; margin-bottom: 20px;">
            🎥 Video Gallery(For 5B)
        </div>
        <tr>
            <td style="width: 25%; vertical-align: top; font-size: 0.9em;">
                <p>A garden comes to life as a kaleidoscope of butterflies flutters amidst the blossoms, their delicate wings casting shadows on the petals below. In the background, a grand fountain cascades water with a gentle splendor, its rhythmic sound providing a soothing backdrop. Beneath the cool shade of a mature tree, a solitary wooden chair invites solitude and reflection, its smooth surface worn by the touch of countless visitors seeking a moment of tranquility in nature's embrace.</p>
            </td>
            <td style="width: 25%; vertical-align: top;">
                <video src="https://github.com/user-attachments/assets/cf5953ea-96d3-48fd-9907-c4708752c714" width="100%" controls autoplay loop></video>
            </td>
            <td style="width: 25%; vertical-align: top; font-size: 0.9em;">
                <p>A small boy, head bowed and determination etched on his face, sprints through the torrential downpour as lightning crackles and thunder rumbles in the distance. The relentless rain pounds the ground, creating a chaotic dance of water droplets that mirror the dramatic sky's anger. In the far background, the silhouette of a cozy home beckons, a faint beacon of safety and warmth amidst the fierce weather. The scene is one of perseverance and the unyielding spirit of a child braving the elements.</p>
            </td>
            <td style="width: 25%; vertical-align: top;">
                <video src="https://github.com/user-attachments/assets/fe0a78e6-b669-4800-8cf0-b5f9b5145b52" width="100%" controls autoplay loop></video>
            </td>
        </tr>
        <tr>
            <td style="width: 25%; vertical-align: top; font-size: 0.9em;">
                <p>A suited astronaut, with the red dust of Mars clinging to their boots, reaches out to shake hands with an alien being, their skin a shimmering blue, under the pink-tinged sky of the fourth planet. In the background, a sleek silver rocket, a beacon of human ingenuity, stands tall, its engines powered down, as the two representatives of different worlds exchange a historic greeting amidst the desolate beauty of the Martian landscape.</p>
            </td>
            <td style="width: 25%; vertical-align: top;">
                <video src="https://github.com/user-attachments/assets/c182f606-8f8c-421d-b414-8487070fcfcb" width="100%" controls autoplay loop></video>
            </td>
            <td style="width: 25%; vertical-align: top; font-size: 0.9em;">
                <p>An elderly gentleman, with a serene expression, sits at the water's edge, a steaming cup of tea by his side. He is engrossed in his artwork, brush in hand, as he renders an oil painting on a canvas that's propped up against a small, weathered table. The sea breeze whispers through his silver hair, gently billowing his loose-fitting white shirt, while the salty air adds an intangible element to his masterpiece in progress. The scene is one of tranquility and inspiration, with the artist's canvas capturing the vibrant hues of the setting sun reflecting off the tranquil sea.</p>
            </td>
            <td style="width: 25%; vertical-align: top;">
                <video src="https://github.com/user-attachments/assets/7db2bbce-194d-434d-a605-350254b6c298" width="100%" controls autoplay loop></video>
            </td>
        </tr>
        <tr>
            <td style="width: 25%; vertical-align: top; font-size: 0.9em;">
                <p>In a dimly lit bar, purplish light bathes the face of a mature man, his eyes blinking thoughtfully as he ponders in close-up, the background artfully blurred to focus on his introspective expression, the ambiance of the bar a mere suggestion of shadows and soft lighting.</p>
            </td>
            <td style="width: 25%; vertical-align: top;">
                <video src="https://github.com/user-attachments/assets/62b01046-8cab-44cc-bd45-4d965bb615ec" width="100%" controls autoplay loop></video>
            </td>
            <td style="width: 25%; vertical-align: top; font-size: 0.9em;">
                <p>A golden retriever, sporting sleek black sunglasses, with its lengthy fur flowing in the breeze, sprints playfully across a rooftop terrace, recently refreshed by a light rain. The scene unfolds from a distance, the dog's energetic bounds growing larger as it approaches the camera, its tail wagging with unrestrained joy, while droplets of water glisten on the concrete behind it. The overcast sky provides a dramatic backdrop, emphasizing the vibrant golden coat of the canine as it dashes towards the viewer.</p>
            </td>
            <td style="width: 25%; vertical-align: top;">
                <video src="https://github.com/user-attachments/assets/d78e552a-4b3f-4b81-ac3f-3898079554f6" width="100%" controls autoplay loop></video>
            </td>
        </tr>
        <tr>
            <td style="width: 25%; vertical-align: top; font-size: 0.9em;">
                <p>On a brilliant sunny day, the lakeshore is lined with an array of willow trees, their slender branches swaying gently in the soft breeze. The tranquil surface of the lake reflects the clear blue sky, while several elegant swans glide gracefully through the still water, leaving behind delicate ripples that disturb the mirror-like quality of the lake. The scene is one of serene beauty, with the willows' greenery providing a picturesque frame for the peaceful avian visitors.</p>
            </td>
            <td style="width: 25%; vertical-align: top;">
                <video src="https://github.com/user-attachments/assets/30894f12-c741-44a2-9e6e-ddcacc231e5b" width="100%" controls autoplay loop></video>
            </td>
            <td style="width: 25%; vertical-align: top; font-size: 0.9em;">
                <p>A Chinese mother, draped in a soft, pastel-colored robe, gently rocks back and forth in a cozy rocking chair positioned in the tranquil setting of a nursery. The dimly lit bedroom is adorned with whimsical mobiles dangling from the ceiling, casting shadows that dance on the walls. Her baby, swaddled in a delicate, patterned blanket, rests against her chest, the child's earlier cries now replaced by contented coos as the mother's soothing voice lulls the little one to sleep. The scent of lavender fills the air, adding to the serene atmosphere, while a warm, orange glow from a nearby nightlight illuminates the scene with a gentle hue, capturing a moment of tender love and comfort.</p>
            </td>
            <td style="width: 25%; vertical-align: top;">
                <video src="https://github.com/user-attachments/assets/926575ca-7150-435b-a0ff-4900a963297b" width="100%" controls autoplay loop></video>
            </td>
        </tr>
    </table>
        """)

    def generate(
        prompt,
        image_input,
        video_input,
        video_strength,
        seed_value,
        scale_status,
        rife_status,
        progress=gr.Progress(track_tqdm=True),
    ):
        # Usa context manager para garantir limpeza de memória ao final
        with memory_manager.temporary_operation("video_generation"):
            try:
                latents, seed, model_path = infer(
                    prompt,
                    image_input,
                    video_input,
                    video_strength,
                    num_inference_steps=50,  # NOT Changed
                    guidance_scale=7.0,  # NOT Changed
                    seed=seed_value,
                    progress=progress,
                )
                
                # Usa context manager para modelos auxiliares (descarrega automaticamente após uso)
                if scale_status:
                    try:
                        with memory_manager.load_model("upscale") as upscale_model:
                            latents = utils.upscale_batch_and_concatenate(upscale_model, latents, device)
                        # Modelo automaticamente descarregado aqui
                    except Exception as e:
                        print(f"⚠️ Upscaling failed: {str(e)}")
                        cleanup_models()
                        raise gr.Error(f"Super-Resolution failed: {str(e)}. Models have been unloaded to free memory.")
                        
                if rife_status:
                    try:
                        with memory_manager.load_model("frame_interpolation") as rife_model:
                            latents = rife_inference_with_latents(rife_model, latents)
                        # Modelo automaticamente descarregado aqui
                    except Exception as e:
                        print(f"⚠️ Frame interpolation failed: {str(e)}")
                        cleanup_models()
                        raise gr.Error(f"Frame Interpolation failed: {str(e)}. Models have been unloaded to free memory.")

                batch_size = latents.shape[0]
                batch_video_frames = []
                for batch_idx in range(batch_size):
                    pt_image = latents[batch_idx]
                    pt_image = torch.stack([pt_image[i] for i in range(pt_image.shape[0])])

                    image_np = VaeImageProcessor.pt_to_numpy(pt_image)
                    image_pil = VaeImageProcessor.numpy_to_pil(image_np)
                    batch_video_frames.append(image_pil)

                # Calculate correct FPS based on model and frame count
                num_frames_generated = len(batch_video_frames[0])
                correct_fps = get_correct_fps(model_path, num_frames_generated)
                
                video_path = utils.save_video(
                    batch_video_frames[0], fps=correct_fps
                )
                video_update = gr.update(visible=True, value=video_path)
                gif_path = convert_to_gif(video_path)
                gif_update = gr.update(visible=True, value=gif_path)
                seed_update = gr.update(visible=True, value=seed)

                return video_path, video_update, gif_update, seed_update
                
            except torch.cuda.OutOfMemoryError as e:
                error_msg = f"GPU Out of Memory: {str(e)}"
                print(f"\n❌ {error_msg}")
                cleanup_on_error()
                raise gr.Error(f"{error_msg}\n\nAll auxiliary models have been unloaded to free memory. Please try again with lower resolution or disable Super-Resolution/Frame Interpolation.")
                
            except RuntimeError as e:
                if "CUDA" in str(e) or "out of memory" in str(e).lower():
                    error_msg = f"CUDA Runtime Error: {str(e)}"
                    print(f"\n❌ {error_msg}")
                    cleanup_on_error()
                    raise gr.Error(f"{error_msg}\n\nModels have been unloaded. Please try again or restart the container if the issue persists.")
                else:
                    error_msg = f"Runtime Error: {str(e)}"
                    print(f"\n❌ {error_msg}")
                    cleanup_on_error()
                    raise gr.Error(error_msg)
                    
            except Exception as e:
                error_msg = f"Unexpected error during generation: {str(e)}"
                print(f"\n❌ {error_msg}")
                print(f"Exception type: {type(e).__name__}")
                import traceback
                traceback.print_exc()
                cleanup_on_error()
                raise gr.Error(f"{error_msg}\n\nModels have been unloaded. Check container logs for details.")
                
            finally:
                # LIMPEZA FORÇADA de memória após cada geração
                print("🧹 Forcing complete memory cleanup after generation...")
                memory_manager.force_cleanup()
                
                # Limpa CUDA cache múltiplas vezes
                if torch.cuda.is_available():
                    for _ in range(3):
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        try:
                            torch.cuda.ipc_collect()
                        except:
                            pass
                
                # Garbage collection agressivo
                import gc
                gc.collect()
                gc.collect()
                
                print("✅ Memory cleanup complete")

    def enhance_prompt_func(prompt):
        return convert_prompt(prompt, retry_times=1)

    generate_button.click(
        generate,
        inputs=[prompt, image_input, video_input, strength, seed_param, enable_scale, enable_rife],
        outputs=[video_output, download_video_button, download_gif_button, seed_text],
    )

    enhance_button.click(enhance_prompt_func, inputs=[prompt], outputs=[prompt])
    video_input.upload(resize_if_unfit, inputs=[video_input], outputs=[video_input])

if __name__ == "__main__":
    demo.queue(max_size=15)
    demo.launch()
