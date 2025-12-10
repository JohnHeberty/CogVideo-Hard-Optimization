"""
THis is the main file for the gradio web demo. It uses the CogVideoX-2B model to generate videos gradio web demo.
set environment variable OPENAI_API_KEY to use the OpenAI API to enhance the prompt.

This demo only supports the text-to-video generation model.
If you wish to use the image-to-video or video-to-video generation models,
please use the gradio_composite_demo to implement the full GUI functionality.

Usage:
    OpenAI_API_KEY=your_openai_api_key OpenAI_BASE_URL=https://api.openai.com/v1 python inference/gradio_web_demo.py
"""

import os
import sys
import threading
import time
import traceback
from pathlib import Path

import gradio as gr
import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video
from datetime import datetime, timedelta
from openai import OpenAI
from moviepy import VideoFileClip

# Add parent directory to path to import fps_utils
sys.path.insert(0, str(Path(__file__).parent))
from fps_utils import get_correct_fps
from vram_utils import get_recommended_offload_strategy, apply_offload_strategy, configure_vae_tiling, get_gpu_memory_info
from logging_config import setup_logging, get_logger, log_generation_params, log_timing, log_vram_status_structured

# Setup logging
logger = setup_logging(log_level=os.getenv("LOG_LEVEL", "INFO"))
logger = get_logger(__name__)

# Model configuration
MODEL_PATH = "THUDM/CogVideoX-5b"

# Global pipeline cache - will be loaded lazily on first use
_pipeline_cache = None


def get_pipeline():
    """
    Lazy load pipeline on first use to avoid loading at startup.
    This saves VRAM when the demo is running but not actively generating.
    
    Returns:
        CogVideoXPipeline configured and ready for inference
        
    Raises:
        RuntimeError: If pipeline loading fails
        torch.cuda.OutOfMemoryError: If insufficient VRAM
    """
    global _pipeline_cache
    
    if _pipeline_cache is not None:
        return _pipeline_cache
    
    try:
        logger.info(f"Loading CogVideoX pipeline from {MODEL_PATH}...")
        log_vram_status_structured(logger)
        
        start_time = time.time()
        pipe = CogVideoXPipeline.from_pretrained(
            MODEL_PATH, torch_dtype=torch.bfloat16
        )
        
        # Apply appropriate offload strategy based on available VRAM
        offload_strategy = get_recommended_offload_strategy(MODEL_PATH)
        apply_offload_strategy(pipe, offload_strategy)
        
        configure_vae_tiling(pipe.vae)
        
        _pipeline_cache = pipe
        
        load_time = time.time() - start_time
        log_timing(logger, "Pipeline Load", load_time)
        logger.info(f"Pipeline loaded successfully with {offload_strategy} offload.")
        log_vram_status_structured(logger)
        
        return _pipeline_cache
        
    except torch.cuda.OutOfMemoryError as e:
        logger.error(f"CUDA Out of Memory while loading pipeline: {e}")
        logger.error("Try closing other programs or using a smaller model.")
        raise RuntimeError(
def convert_prompt(prompt: str, retry_times: int = 3) -> str:
    """
    Enhance prompt using OpenAI API (GLM-4 model).
    
    Args:
        prompt: Original prompt
        retry_times: Number of retries if API fails
        
    Returns:
        Enhanced prompt, or original if API unavailable or fails
    """
    if not os.environ.get("OPENAI_API_KEY"):
        logger.debug("OPENAI_API_KEY not set, skipping prompt enhancement")
        return prompt

    try:
        client = OpenAI()
        text = prompt.strip()

        for i in range(retry_times):
            try:
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
                    max_tokens=250,
                )
                if response.choices:
                    enhanced = response.choices[0].message.content
                    logger.info(f"Prompt enhanced successfully (attempt {i+1}/{retry_times})")
                    return enhanced
            except Exception as e:
                logger.warning(f"Prompt enhancement attempt {i+1}/{retry_times} failed: {e}")
                if i < retry_times - 1:
                    time.sleep(1)  # Wait before retry
                continue
                
        logger.warning("All prompt enhancement attempts failed, using original prompt")
        return prompt
        
    except Exception as e:
        logger.error(f"Prompt enhancement failed: {e}")
        logger.debug(traceback.format_exc())
        return prompt
def infer(
    prompt: str,
    num_inference_steps: int,
    guidance_scale: float,
    progress=gr.Progress(track_tqdm=True),
):
    """
    Run video generation inference.
    
    Args:
        prompt: Text prompt for generation
        num_inference_steps: Number of denoising steps
        guidance_scale: Classifier-free guidance scale
        progress: Gradio progress tracker
        
    Returns:
        Generated video tensor
        
    Raises:
def save_video(tensor):
    """
    Save video tensor to file.
    
    Args:
        tensor: Video tensor to save
        
    Returns:
        Path to saved video file
        
    Raises:
        gr.Error: If save fails
    """
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = f"./output/{timestamp}.mp4"
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        
        # Auto-detect correct FPS based on model and frame count
        num_frames = len(tensor) if isinstance(tensor, list) else tensor.shape[0]
        fps = get_correct_fps(MODEL_PATH, num_frames)
        
        logger.info(f"Saving video to {video_path} ({num_frames} frames @ {fps}fps)")
        start_time = time.time()
        
        export_to_video(tensor, video_path, fps=fps)
        
        save_time = time.time() - start_time
        log_timing(logger, "Video Save", save_time)
        
        # Verify file was created
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file was not created: {video_path}")
        
        file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
        logger.info(f"Video saved successfully: {file_size_mb:.1f}MB")
        
        return video_path
        
    except FileNotFoundError as e:
def convert_to_gif(video_path):
    """
    Convert MP4 video to GIF.
    
    Args:
        video_path: Path to MP4 video
        
    Returns:
        Path to generated GIF
        
    Raises:
        gr.Error: If conversion fails
    """
    try:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        logger.info(f"Converting video to GIF: {video_path}")
        start_time = time.time()
        
        clip = VideoFileClip(video_path)
        clip = clip.with_fps(8)
        clip = clip.resized(height=240)
        gif_path = video_path.replace(".mp4", ".gif")
        clip.write_gif(gif_path, fps=8)
        clip.close()
        
        conv_time = time.time() - start_time
        log_timing(logger, "GIF Conversion", conv_time)
        
        if not os.path.exists(gif_path):
            raise FileNotFoundError(f"GIF file was not created: {gif_path}")
        
        gif_size_mb = os.path.getsize(gif_path) / (1024 * 1024)
        logger.info(f"GIF created successfully: {gif_size_mb:.1f}MB")
        
        return gif_path
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise gr.Error(f"❌ Video file not found for GIF conversion")
    except Exception as e:
        logger.error(f"GIF conversion failed: {e}")
        logger.error(traceback.format_exc())
        raise gr.Error(f"❌ Failed to create GIF: {str(e)}")r(f"Failed to save video: {e}")
        logger.error(traceback.format_exc())
        raise gr.Error(f"❌ Failed to save video: {str(e)}")nce_steps < 10 or num_inference_steps > 100:
            raise gr.Error("❌ Inference steps must be between 10 and 100. Recommended: 50 steps.")
        
        if guidance_scale < 1.0 or guidance_scale > 20.0:
            raise gr.Error("❌ Guidance scale must be between 1.0 and 20.0. Recommended: 6.0")
        
        # Log generation parameters
        log_generation_params(
            logger,
            prompt=prompt,
            model_path=MODEL_PATH,
            num_frames=49,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        )
        
        # Load pipeline (may fail if VRAM insufficient)
        pipe = get_pipeline()
        
        # Clear VRAM before generation
        torch.cuda.empty_cache()
        log_vram_status_structured(logger)
        
        # Run generation
        logger.info("Starting video generation...")
        start_time = time.time()
        
        video = pipe(
            prompt=prompt,
            num_videos_per_prompt=1,
            num_inference_steps=num_inference_steps,
            num_frames=49,
            guidance_scale=guidance_scale,
        ).frames[0]
        
        gen_time = time.time() - start_time
        log_timing(logger, "Video Generation", gen_time)
        log_vram_status_structured(logger)
        
        logger.info("Video generation completed successfully!")
        return video
        
    except torch.cuda.OutOfMemoryError as e:
        logger.error(f"CUDA OOM during generation: {e}")
        torch.cuda.empty_cache()
        raise gr.Error(
            "❌ Out of VRAM! Try:\n"
            "• Use CogVideoX-2b instead of 5b\n"
            "• Reduce inference steps (try 30 instead of 50)\n"
            "• Close other GPU applications\n"
            "• Enable model offloading in settings"
        )
    except ValueError as e:
        logger.error(f"Invalid parameter: {e}")
        raise gr.Error(f"❌ Invalid parameter: {str(e)}")
    except RuntimeError as e:
        logger.error(f"Runtime error during generation: {e}")
        logger.error(traceback.format_exc())
        raise gr.Error(f"❌ Generation failed: {str(e)}\n\nCheck console logs for details.")
    except Exception as e:
        logger.error(f"Unexpected error during generation: {e}")
        logger.error(traceback.format_exc())
        raise gr.Error(f"❌ Unexpected error: {str(e)}\n\nPlease try again or check console logs.")l="glm-4-plus",
            temperature=0.01,
            top_p=0.7,
            stream=False,
            max_tokens=250,
        )
        if response.choices:
            return response.choices[0].message.content
    return prompt


def infer(
    prompt: str,
    num_inference_steps: int,
    guidance_scale: float,
    progress=gr.Progress(track_tqdm=True),
):
    pipe = get_pipeline()
    torch.cuda.empty_cache()
    video = pipe(
        prompt=prompt,
        num_videos_per_prompt=1,
        num_inference_steps=num_inference_steps,
        num_frames=49,
        guidance_scale=guidance_scale,
    ).frames[0]

    return video

    def generate(
        prompt,
        num_inference_steps,
        guidance_scale,
        progress=gr.Progress(track_tqdm=True),
    ):
        """
        Main generation function with comprehensive error handling.
        
        All exceptions are caught and converted to gr.Error for user-friendly display.
        """
        try:
            tensor = infer(prompt, num_inference_steps, guidance_scale, progress=progress)
            video_path = save_video(tensor)
            video_update = gr.update(visible=True, value=video_path)
            gif_path = convert_to_gif(video_path)
            gif_update = gr.update(visible=True, value=gif_path)

            return video_path, video_update, gif_update
            
        except gr.Error:
            # Re-raise Gradio errors (already user-friendly)
            raise
        except Exception as e:
            # Catch any unhandled exceptions
            logger.error(f"Unhandled error in generate(): {e}")
            logger.error(traceback.format_exc())
            raise gr.Error(f"❌ Generation failed: {str(e)}\n\nPlease check console logs for details.")

    def enhance_prompt_func(prompt):
        """
        Enhance prompt with error handling.
        """
        try:
            if not prompt or len(prompt.strip()) == 0:
                raise gr.Error("❌ Cannot enhance empty prompt. Please enter a description first.")
            
            return convert_prompt(prompt, retry_times=1)
            
        except gr.Error:
            raise
        except Exception as e:
            logger.error(f"Prompt enhancement failed: {e}")
            logger.error(traceback.format_exc())
            raise gr.Error(f"❌ Prompt enhancement failed: {str(e)}\n\nUsing original prompt.")
    clip = clip.resized(height=240)
    gif_path = video_path.replace(".mp4", ".gif")
    clip.write_gif(gif_path, fps=8)
    return gif_path


def delete_old_files():
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
        time.sleep(600)


threading.Thread(target=delete_old_files, daemon=True).start()

with gr.Blocks() as demo:
    gr.Markdown("""
           <div style="text-align: center; font-size: 32px; font-weight: bold; margin-bottom: 20px;">
               CogVideoX Gradio Simple Space🤗
            """)

    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(
                label="Prompt (Less than 200 Words)", placeholder="Enter your prompt here", lines=5
            )

            with gr.Row():
                gr.Markdown(
                    "✨Upon pressing the enhanced prompt button, we will use [GLM-4 Model](https://github.com/THUDM/GLM-4) to polish the prompt and overwrite the original one."
                )
                enhance_button = gr.Button("✨ Enhance Prompt(Optional)")

            with gr.Column():
                gr.Markdown(
                    "**Optional Parameters** (default values are recommended)<br>"
                    "Increasing the number of inference steps will produce more detailed videos, but it will slow down the process.<br>"
                    "50 steps are recommended for most cases.<br>"
                )
                with gr.Row():
                    num_inference_steps = gr.Number(label="Inference Steps", value=50)
                    guidance_scale = gr.Number(label="Guidance Scale", value=6.0)
                generate_button = gr.Button("🎬 Generate Video")

        with gr.Column():
            video_output = gr.Video(label="CogVideoX Generate Video", width=720, height=480)
            with gr.Row():
                download_video_button = gr.File(label="📥 Download Video", visible=False)
                download_gif_button = gr.File(label="📥 Download GIF", visible=False)

    def generate(
        prompt,
        num_inference_steps,
        guidance_scale,
        model_choice,
        progress=gr.Progress(track_tqdm=True),
    ):
        tensor = infer(prompt, num_inference_steps, guidance_scale, progress=progress)
        video_path = save_video(tensor)
        video_update = gr.update(visible=True, value=video_path)
        gif_path = convert_to_gif(video_path)
        gif_update = gr.update(visible=True, value=gif_path)

        return video_path, video_update, gif_update

    def enhance_prompt_func(prompt):
        return convert_prompt(prompt, retry_times=1)

    generate_button.click(
        generate,
        inputs=[prompt, num_inference_steps, guidance_scale],
        outputs=[video_output, download_video_button, download_gif_button],
    )

    enhance_button.click(enhance_prompt_func, inputs=[prompt], outputs=[prompt])

if __name__ == "__main__":
    demo.launch()
