from diffusers import StableDiffusionXLPipeline
import torch

model_id = "/home/jenyuan/zoo/stable-diffusion-xl-base-1.0"
pipe = StableDiffusionXLPipeline.from_pretrained(model_id,torch_dtype=torch.float16).to("cuda")

pipe.load_textual_inversion(
    "./dog_fixed/learned_embeds-steps-1000.safetensors",
    tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder
)

pipe.load_textual_inversion(
    "./dog_fixed/learned_embeds_2-steps-1000.safetensors",
    tokenizer=pipe.tokenizer_2, text_encoder=pipe.text_encoder_2
)
pipe.text_encoder.text_model.eos_token_id = pipe.tokenizer.eos_token_id
pipe.text_encoder_2.text_model.eos_token_id = pipe.tokenizer_2.eos_token_id

prompt = "A photo of a <object>"

generator = torch.Generator(device=pipe.device).manual_seed(42)
image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5, generator=generator).images[0]

image.save("dog.png")