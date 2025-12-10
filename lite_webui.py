import os

import gradio as gr
import torch
from diffusers import (
    CogVideoXPipeline,
    CogVideoXImageToVideoPipeline,
    CogVideoXDPMScheduler,
)
import utils  # utils.py do próprio projeto (mesma pasta do script original)

# Define device global
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"[GLOBAL] torch.cuda.is_available() = {torch.cuda.is_available()}")
print(f"[GLOBAL] DEVICE = {DEVICE}", flush=True)

# Modelos configuráveis via variáveis de ambiente
T2V_MODEL_ID = os.environ.get("COGVIDEO_T2V_MODEL", "THUDM/CogVideoX-2b")
I2V_MODEL_ID = os.environ.get("COGVIDEO_I2V_MODEL", "THUDM/CogVideoX-5b-I2V")


def _save_video(frames, fps: int = 8) -> str:
    """Salva frames em vídeo usando a função oficial do projeto."""
    video_path = utils.save_video(frames, fps=fps)
    return video_path


# -----------------------------
# Text → Video (lazy load)
# -----------------------------
def generate_text_to_video(prompt, num_steps, guidance):
    """
    Text -> Video com lazy load:
      - Carrega CogVideoX-2B dentro da função.
      - Gera o vídeo.
      - Deleta o pipeline e limpa VRAM no final.
    """
    if not prompt or not prompt.strip():
        raise gr.Error("Digite um prompt de texto primeiro.")

    print("[T2V] Iniciando geração...", flush=True)
    print(f"[T2V] DEVICE = {DEVICE}", flush=True)

    pipe = None
    try:
        dtype = torch.float16 if DEVICE.type == "cuda" else torch.float32

        pipe = CogVideoXPipeline.from_pretrained(
            T2V_MODEL_ID,
            torch_dtype=dtype,
        )

        pipe.scheduler = CogVideoXDPMScheduler.from_config(
            pipe.scheduler.config,
            timestep_spacing="trailing",
        )

        # Reduz pico de memória no decode do VAE
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()

        pipe.to(DEVICE)
        print("[T2V] Pipeline carregado.", flush=True)

        result = pipe(
            prompt=prompt,
            num_inference_steps=int(num_steps),
            guidance_scale=float(guidance),
        )

        frames = result.frames[0]
        video_path = _save_video(frames, fps=8)
        print("[T2V] Geração concluída.", flush=True)
        return video_path

    finally:
        print("[T2V] Liberando modelo da memória...", flush=True)
        if pipe is not None:
            del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# -----------------------------
# Image → Video (lazy load)
# -----------------------------
def generate_image_to_video(image, prompt, num_frames, num_steps, guidance):
    """
    Image -> Video com lazy load:
      - Carrega CogVideoX-5b-I2V dentro da função.
      - Usa model_cpu_offload se possível (para caber melhor em 24GB).
      - Deleta o pipeline e limpa VRAM no final.
    """
    if image is None:
        raise gr.Error("Envie uma imagem primeiro.")

    print("[I2V] Iniciando geração...", flush=True)
    print(f"[I2V] DEVICE = {DEVICE}", flush=True)

    pipe = None
    try:
        dtype = torch.bfloat16 if DEVICE.type == "cuda" else torch.float32

        pipe = CogVideoXImageToVideoPipeline.from_pretrained(
            I2V_MODEL_ID,
            torch_dtype=dtype,
        )

        # Otimizações de VAE para reduzir pico de VRAM na decodificação
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()

        if DEVICE.type == "cuda":
            try:
                pipe.enable_model_cpu_offload()
                print("[I2V] enable_model_cpu_offload() ativado.", flush=True)
            except Exception as e:
                print(f"[I2V] Falha em enable_model_cpu_offload(): {e}", flush=True)
                pipe.to(DEVICE)
        else:
            pipe.to(DEVICE)

        print("[I2V] Pipeline carregado.", flush=True)

        result = pipe(
            image=image,
            prompt=prompt or "",
            num_frames=int(num_frames),
            num_inference_steps=int(num_steps),
            guidance_scale=float(guidance),
        )

        frames = result.frames[0]
        video_path = _save_video(frames, fps=8)
        print("[I2V] Geração concluída.", flush=True)
        return video_path

    finally:
        print("[I2V] Liberando modelo da memória...", flush=True)
        if pipe is not None:
            del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# -----------------------------
# UI Gradio
# -----------------------------
def build_ui():
    with gr.Blocks() as demo:
        gr.Markdown(
            "# CogVideoX Lite UI\n"
            "- **Lazy load total**: Text→Video e Image→Video só carregam o modelo na VRAM quando você clica em gerar.\n"
            "- Depois de gerar, o modelo é descarregado e a VRAM é limpa (dentro do possível).\n"
            f"- Device detectado: `{DEVICE}`\n"
            "- Text→Video usa `CogVideoX-2B`; Image→Video usa `CogVideoX-5b-I2V`."
        )

        # --- Aba Text -> Video ---
        with gr.Tab("Text → Video"):
            with gr.Row():
                with gr.Column(scale=1):
                    prompt_t = gr.Textbox(
                        label="Prompt",
                        lines=6,
                        placeholder=(
                            "A panda playing guitar in a bamboo forest at sunset, "
                            "cinematic, 4k..."
                        ),
                    )
                    steps_t = gr.Slider(
                        label="Inference steps",
                        minimum=20,
                        maximum=80,
                        step=5,
                        value=40,
                    )
                    guidance_t = gr.Slider(
                        label="Guidance scale",
                        minimum=1.0,
                        maximum=10.0,
                        step=0.5,
                        value=6.0,
                    )
                    btn_t = gr.Button("🎬 Gerar vídeo (Texto → Vídeo)")

                with gr.Column(scale=1):
                    out_video_t = gr.Video(label="Saída (Text→Video)")

            btn_t.click(
                generate_text_to_video,
                inputs=[prompt_t, steps_t, guidance_t],
                outputs=[out_video_t],
            )

        # --- Aba Image -> Video ---
        with gr.Tab("Image → Video"):
            with gr.Row():
                with gr.Column(scale=1):
                    img_i = gr.Image(label="Imagem de entrada", type="pil")
                    prompt_i = gr.Textbox(
                        label="Prompt (opcional, ajuda na direção do movimento)",
                        lines=4,
                        placeholder="Camera orbiting around the subject...",
                    )
                    num_frames_i = gr.Slider(
                        label="Nº de frames",
                        minimum=16,
                        maximum=40,
                        step=1,
                        value=24,
                    )
                    steps_i = gr.Slider(
                        label="Inference steps",
                        minimum=20,
                        maximum=80,
                        step=5,
                        value=40,
                    )
                    guidance_i = gr.Slider(
                        label="Guidance scale",
                        minimum=1.0,
                        maximum=10.0,
                        step=0.5,
                        value=6.0,
                    )
                    btn_i = gr.Button("🎬 Gerar vídeo (Imagem → Vídeo)")

                with gr.Column(scale=1):
                    out_video_i = gr.Video(label="Saída (Image→Video)")

            btn_i.click(
                generate_image_to_video,
                inputs=[img_i, prompt_i, num_frames_i, steps_i, guidance_i],
                outputs=[out_video_i],
            )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.queue(max_size=2)
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
    )

