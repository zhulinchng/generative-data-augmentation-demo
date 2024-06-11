import random

import gradio as gr
import numpy as np
import torch
import torchvision.transforms as transforms
from torchmetrics.functional.image import structural_similarity_index_measure as ssim
from transformers import CLIPModel, CLIPProcessor

from tools import synth

device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "runwayml/stable-diffusion-v1-5"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

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
        apply_optimization=False,
    )

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 1024


def infer(
    input_image,
    prompt1,
    prompt2,
    negative_prompt,
    seed,
    randomize_seed,
    width,
    height,
    guidance_scale,
    interpolation_step,
    num_inference_steps,
    num_interpolation_steps,
    sample_mid_interpolation,
    remove_n_middle,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    prompts = [prompt1, prompt2]
    generator = torch.Generator().manual_seed(seed)
    print(seed)
    interpolated_prompt_embeds, prompt_metadata = synth.interpolatePrompts(
        prompts,
        pipe,
        num_interpolation_steps,
        sample_mid_interpolation,
        remove_n_middle=remove_n_middle,
        device=device,
    )
    negative_prompts = [negative_prompt, negative_prompt]
    if negative_prompts != ["", ""]:
        interpolated_negative_prompts_embeds, negative_prompt_metadata = (
            synth.interpolatePrompts(
                negative_prompts,
                pipe,
                num_interpolation_steps,
                sample_mid_interpolation,
                remove_n_middle=remove_n_middle,
                device=device,
            )
        )
    else:
        interpolated_negative_prompts_embeds, negative_prompt_metadata = [None] * len(
            interpolated_prompt_embeds
        ), None

    latents = torch.randn(
        (1, pipe.unet.config.in_channels, height // 8, width // 8),
        generator=generator,
    ).to(device)
    embed_pairs = zip(interpolated_prompt_embeds, interpolated_negative_prompts_embeds)
    embed_pairs_list = list(embed_pairs)
    print(len(embed_pairs_list))
    # offset step by -1
    prompt_embeds, negative_prompt_embeds = embed_pairs_list[interpolation_step - 1]
    preprocess_input = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize((512, 512))]
    )
    input_img_tensor = preprocess_input(input_image).unsqueeze(0)
    if negative_prompt_embeds is not None:
        npe = negative_prompt_embeds[None, ...]
    else:
        npe = None
    image = pipe(
        height=height,
        width=width,
        num_images_per_prompt=1,
        prompt_embeds=prompt_embeds[None, ...],
        negative_prompt_embeds=npe,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        latents=latents,
        image=input_img_tensor,
    ).images[0]
    pred_image = transforms.ToTensor()(image).unsqueeze(0)
    ssim_score = ssim(pred_image, input_img_tensor).item()
    real_inputs = clip_processor(
        text=prompts, padding=True, images=input_image, return_tensors="pt"
    ).to(device)
    real_output = clip_model(**real_inputs)
    synth_inputs = clip_processor(
        text=prompts, padding=True, images=image, return_tensors="pt"
    ).to(device)
    synth_output = clip_model(**synth_inputs)
    cos_sim = torch.nn.CosineSimilarity(dim=1)
    cosine_sim = (
        cos_sim(real_output.image_embeds, synth_output.image_embeds)
        .detach()
        .cpu()
        .numpy()
        .squeeze()
        * 100
    )

    return image, seed, ssim_score, cosine_sim


examples1 = [
    "A photo of a chain saw, chainsaw",
    "A photo of a Shih-Tzu, a type of dog",
]
examples2 = [
    "A photo of a golf ball",
    "A photo of a beagle, a type of dog",
]

css = """
#col-container {
    margin: 0 auto;
}
"""


def update_steps(total_steps, interpolation_step):
    if interpolation_step > total_steps:
        return gr.update(maximum=total_steps // 2, value=total_steps)
    return gr.update(maximum=total_steps // 2)


if torch.cuda.is_available():
    power_device = "GPU"
else:
    power_device = "CPU"

with gr.Blocks(css=css, title="Generative Date Augmentation") as demo:

    with gr.Column(elem_id="col-container"):
        gr.Markdown(
            f"""
        # Data Augmentation with Image-to-Image Diffusion Models via Prompt Interpolation 
        Currently running on {power_device}.
        """
        )

        input_image = gr.Image(type="pil", label="Image to Augment")

        with gr.Row():
            prompt1 = gr.Text(
                label="Prompt 1",
                show_label=True,
                max_lines=1,
                placeholder="Enter your first prompt",
                container=False,
            )
        with gr.Row():
            prompt2 = gr.Text(
                label="Prompt 2",
                show_label=True,
                max_lines=1,
                placeholder="Enter your second prompt",
                container=False,
            )
        with gr.Row():
            gr.Examples(
                examples=examples1, inputs=[prompt1], label="Example for Prompt 1"
            )
            gr.Examples(
                examples=examples2, inputs=[prompt2], label="Example for Prompt 2"
            )

        with gr.Row():
            num_interpolation_steps = gr.Slider(
                label="Total interpolation steps",
                minimum=2,
                maximum=32,
                step=2,
                value=16,
            )
            interpolation_step = gr.Slider(
                label="Specific Interpolation Step",
                minimum=1,
                maximum=8,
                step=1,
                value=8,
            )
            num_interpolation_steps.change(
                fn=update_steps,
                inputs=[num_interpolation_steps, interpolation_step],
                outputs=[interpolation_step],
            )
            run_button = gr.Button("Run", scale=0)

        result = gr.Image(label="Result", show_label=False)

        with gr.Accordion("Advanced Settings", open=True):

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

            gr.Markdown("Negative Prompt: ")
            with gr.Row():
                negative_prompt = gr.Text(
                    label="Negative Prompt",
                    show_label=True,
                    max_lines=1,
                    value="blurry image, disfigured, deformed, distorted, cartoon, drawings",
                    container=False,
                )
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
                    value=8.0,
                )

                num_inference_steps = gr.Slider(
                    label="Number of inference steps",
                    minimum=1,
                    maximum=80,
                    step=1,
                    value=25,
                )
            with gr.Row():
                sample_mid_interpolation = gr.Slider(
                    label="Number of sampling steps in the middle of interpolation",
                    minimum=2,
                    maximum=80,
                    step=2,
                    value=16,
                )
            with gr.Row():
                remove_n_middle = gr.Slider(
                    label="Number of middle steps to remove from interpolation",
                    minimum=0,
                    maximum=80,
                    step=2,
                    value=0,
                )
        gr.Markdown(
            """
            Metadata:
            """
        )
        with gr.Row():
            show_seed = gr.Label(label="Seed:", value="Randomized seed")
            ssim_score = gr.Label(label="SSIM Score:", value="Generate to see score")
            cos_sim = gr.Label(label="CLIP Score:", value="Generate to see score")
    run_button.click(
        fn=infer,
        inputs=[
            input_image,
            prompt1,
            prompt2,
            negative_prompt,
            seed,
            randomize_seed,
            width,
            height,
            guidance_scale,
            interpolation_step,
            num_inference_steps,
            num_interpolation_steps,
            sample_mid_interpolation,
            remove_n_middle,
        ],
        outputs=[result, show_seed, ssim_score, cos_sim],
    )

demo.queue().launch()

"""
    input_image,
    prompt1,
    prompt2,
    negative_prompt,
    seed,
    randomize_seed,
    width,
    height,
    guidance_scale,
    interpolation_step,
    num_inference_steps,
    num_interpolation_steps,
    sample_mid_interpolation,
    remove_n_middle,
    """
