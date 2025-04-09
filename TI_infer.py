from diffusers import StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer
import torch
import safetensors
import os

model_id = "/apdcephfs/default121254/apdcephfs_cq10/share_916081/jentsehuang/models/bk-sdm-v2-small"
# model_id = "/apdcephfs/default121133/apdcephfs_qy3/share_301812049/jentsehuang/models/stable-diffusion-2-1-base"
pipe = StableDiffusionPipeline.from_pretrained(model_id,torch_dtype=torch.float16).to("cuda")
seed=1017
distilled_ckpt = "/apdcephfs/default121254/apdcephfs_cq8/share_916081/jentsehuang/textual_inversion/output/full/baseline/backpack"
prompt = "a photo of a <object>"
out_path = f"{distilled_ckpt}/samples"
learned_token_path = f"{distilled_ckpt}"
os.makedirs(out_path, exist_ok=True)
sample_prefix = "sd_token_bk_model"

# Load distilled weights
# state_dict = torch.load(f"{distilled_ckpt}/ckpts/checkpoint-30000/model_ckpt.pt")
# state_dict = dict()
# with safetensors.safe_open(f"/apdcephfs/default092611/apdcephfs_qy3/share_301812049/jentsehuang/embd_GM_cos_5A100/ckpts/checkpoint-40000/model_1.safetensors", framework="pt", device="cpu") as f:
#     for key in f.keys():
#         state_dict[key] = f.get_tensor(key)
# pipe.unet.load_state_dict(state_dict)

token_ckpt = "learned_embeds-steps-500.safetensors"
repo_id_embeds = os.path.join(learned_token_path, token_ckpt)
pipe.load_textual_inversion(repo_id_embeds)
input_ids = pipe.tokenizer(
    [prompt], max_length=pipe.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
).input_ids.to("cuda")
txt_emb_500 = pipe.text_encoder(input_ids)[0]
generator = torch.Generator(device=pipe.device).manual_seed(seed)
image = pipe(prompt_embeds=txt_emb_500, num_inference_steps=50, guidance_scale=7.5, generator=generator).images[0]
image.save(f"{out_path}/{sample_prefix}_500.png")
pipe.unload_textual_inversion()

token_ckpt = "learned_embeds-steps-1000.safetensors"
repo_id_embeds = os.path.join(learned_token_path, token_ckpt)
pipe.load_textual_inversion(repo_id_embeds)
input_ids = pipe.tokenizer(
    [prompt], max_length=pipe.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
).input_ids.to("cuda")
txt_emb_1000 = pipe.text_encoder(input_ids)[0]
cos_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)(txt_emb_500.view(1,-1), txt_emb_1000.view(1,-1))
generator = torch.Generator(device=pipe.device).manual_seed(seed)
image = pipe(prompt_embeds=txt_emb_1000, num_inference_steps=50, guidance_scale=7.5, generator=generator).images[0]
image.save(f"{out_path}/{sample_prefix}_1000.png")
pipe.unload_textual_inversion()
print(txt_emb_500.shape, txt_emb_1000.shape, cos_sim)
