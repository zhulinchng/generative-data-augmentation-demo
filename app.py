import random

import gradio as gr
import numpy as np
import torch

from tools import synth

device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "runwayml/stable-diffusion-v1-5"

if torch.cuda.is_available():
    torch.cuda.max_memory_allocated(device=device)
    pipe = synth.pipe_img(
        model_path=model_path,
        device=device,
        use_torchcompile=False,
    )
else:
    pipe = synth.pipe_img(
        model_path=model_path,
        device=device,
        use_torchcompile=False,
    )

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 1024


def infer(
    input_image,
    prompt,
    negative_prompt,
    seed,
    randomize_seed,
    width,
    height,
    guidance_scale,
    num_inference_steps,
):

    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    generator = torch.Generator().manual_seed(seed)

    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        width=width,
        height=height,
        generator=generator,
        image=input_image,
    ).images[0]

    return image


examples = [
    "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
    "An astronaut riding a green horse",
    "A delicious ceviche cheesecake slice",
]

css = """
#col-container {
    margin: 0 auto;
    max-width: 520px;
}
"""

if torch.cuda.is_available():
    power_device = "GPU"
else:
    power_device = "CPU"

with gr.Blocks(css=css) as demo:

    with gr.Column(elem_id="col-container"):
        gr.Markdown(
            f"""
        # Text-to-Image Gradio Template
        Currently running on {power_device}.
        """
        )

        with gr.Row():

            prompt = gr.Text(
                label="Prompt",
                show_label=False,
                max_lines=1,
                placeholder="Enter your prompt",
                container=False,
            )
            input_image = gr.Image(type="pil", label="Input Image")

            run_button = gr.Button("Run", scale=0)

        result = gr.Image(label="Result", show_label=False)

        with gr.Accordion("Advanced Settings", open=False):

            negative_prompt = gr.Text(
                label="Negative prompt",
                max_lines=1,
                placeholder="Enter a negative prompt",
                visible=False,
            )

            seed = gr.Slider(
                label="Seed",
                minimum=0,
                maximum=MAX_SEED,
                step=1,
                value=0,
            )

            randomize_seed = gr.Checkbox(label="Randomize seed", value=True)

            with gr.Row():

                width = gr.Slider(
                    label="Width",
                    minimum=256,
                    maximum=MAX_IMAGE_SIZE,
                    step=32,
                    value=512,
                )

                height = gr.Slider(
                    label="Height",
                    minimum=256,
                    maximum=MAX_IMAGE_SIZE,
                    step=32,
                    value=512,
                )

            with gr.Row():

                guidance_scale = gr.Slider(
                    label="Guidance scale",
                    minimum=0.0,
                    maximum=10.0,
                    step=0.1,
                    value=0.0,
                )

                num_inference_steps = gr.Slider(
                    label="Number of inference steps",
                    minimum=1,
                    maximum=12,
                    step=1,
                    value=2,
                )

        gr.Examples(examples=examples, inputs=[prompt])

    run_button.click(
        fn=infer,
        inputs=[
            input_image,
            prompt,
            negative_prompt,
            seed,
            randomize_seed,
            width,
            height,
            guidance_scale,
            num_inference_steps,
        ],
        outputs=[result],
    )

demo.queue().launch()
