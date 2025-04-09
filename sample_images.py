from diffusers import StableDiffusionPipeline
import torch
import os
import argparse


def main(args):
    model_id = "/apdcephfs_cq10/share_916081/jentsehuang/models/stable-diffusion-v1-4"
    pipe = StableDiffusionPipeline.from_pretrained(model_id,torch_dtype=torch.float16).to("cuda")
    test_path = args.test_path
    out_path = args.out_path
    prompt = "a photo of a <object>"

    for cls_label in os.listdir(test_path):
        os.makedirs(os.path.join(out_path, cls_label), exist_ok=True)
        learned_token_path = os.path.join(test_path, cls_label, "learned_embeds-steps-1000.safetensors")
        pipe.load_textual_inversion(learned_token_path)
        input_ids = pipe.tokenizer(
            [prompt], max_length=pipe.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids.to("cuda")
        learned_txt_emb = pipe.text_encoder(input_ids)[0]
        pipe.unload_textual_inversion()

        for j in range(64):
            generator = torch.Generator(device=pipe.device).manual_seed(3467+j)
            image = pipe(prompt_embeds=learned_txt_emb, num_inference_steps=50, guidance_scale=7.5, generator=generator).images[0]
            image.save(os.path.join(out_path, cls_label, f"{j}.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', required=True)
    parser.add_argument('--out_path', required=True)
    args = parser.parse_args()

    main(args)
