from diffusers import StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer, AutoProcessor, CLIPVisionModel, AutoImageProcessor, AutoModel
import torch
import os
import numpy as np
import time
from imagenet1k_classes import IMAGENET2012_CLASSES
from PIL import Image
import argparse
import pyiqa
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import tqdm
from torch.utils.data import Dataset
from torchvision import transforms


class PairedDataset(Dataset):
    def __init__(self, gt_path, test_path):
        self.test_path = test_path
        self.gt_path = gt_path
        self.cls = os.listdir(os.path.join(gt_path))
        self.image_num = len(os.listdir(os.path.join(test_path, self.cls[0])))

    def __len__(self):
        return self.image_num * len(self.cls)

    def __getitem__(self, idx):
        cls_folder = self.cls[idx // len(self.cls)]
        gt_image = os.listdir(os.path.join(self.gt_path, cls_folder))[0]
        return { "test_image": os.path.join(self.test_path, cls_folder, f"{idx % self.image_num}.png"),
                 "gt_image": os.path.join(self.gt_path, cls_folder, gt_image)}

def main(args):

    gt_path = args.gt_path
    test_path = args.test_path

    # embedding models
    print(f"Loading models...")
    if args.clip_image_emb:
        dataset = PairedDataset(gt_path, test_path)
        model_id = args.clip_path
        clip_model = CLIPVisionModel.from_pretrained(model_id).to("cuda", dtype=torch.float32)
        processor = AutoProcessor.from_pretrained(model_id)
        # clip_model = AutoModel.from_pretrained(model_id).to("cuda", dtype=torch.float32)
        # processor = AutoImageProcessor.from_pretrained(model_id)
        def collate_fn(examples):
            test_images = []
            for example in examples:
                test_img = Image.open(example["test_image"])
                test_images.append(processor(images=test_img, return_tensors="pt")['pixel_values'])
            test_images = torch.cat(test_images, dim=0)
            test_images = test_images.to(memory_format=torch.contiguous_format).float()
            gt_image = Image.open(examples[0]["gt_image"])
            gt_image = processor(images=gt_image, return_tensors="pt")['pixel_values']
            gt_image = gt_image.to(memory_format=torch.contiguous_format).float()
            return {"test_image": test_images,
                    "gt_image": gt_image}
        test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=16, pin_memory=True, drop_last=False, collate_fn=collate_fn)
    elif args.iqa_test:
        dataset = PairedDataset(gt_path, test_path)
        fid_metric = pyiqa.create_metric('lpips', device=torch.device('cuda'))
        pt_transform = transforms.Compose([transforms.Resize((512,512)), transforms.ToTensor()])
        def collate_fn(examples):
            test_images = []
            for example in examples:
                test_img = Image.open(example["test_image"])
                test_images.append(pt_transform(test_img).unsqueeze(0))
            test_images = torch.cat(test_images, dim=0)
            test_images = test_images.to(memory_format=torch.contiguous_format).float()
            gt_image = Image.open(examples[0]["gt_image"])
            gt_image = pt_transform(gt_image).unsqueeze(0)
            gt_image = gt_image.to(memory_format=torch.contiguous_format).float()
            return {"test_image": test_images,
                    "gt_image": gt_image}
        test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, num_workers=16, pin_memory=True, drop_last=False, collate_fn=collate_fn)
    elif args.clip_text_emb:
        # model_id = "/apdcephfs/default121133/apdcephfs_qy3/share_301812049/jentsehuang/models/stable-diffusion-2-1-base"
        model_id = args.sd_path
        pipe = StableDiffusionPipeline.from_pretrained(model_id,torch_dtype=torch.float32).to("cuda")
        prompt = "a photo of a <object>"

    # metrics
    CosineSim = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

    if args.imagenet_cls:
        embedding_dict = list()
        index_dict = dict()
        with torch.no_grad():
            for i, (k,v) in enumerate(IMAGENET2012_CLASSES.items()):
                input_ids = pipe.tokenizer(
                    [f"a photo of a {v}"], max_length=pipe.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
                ).input_ids.to("cuda")
                cls_emb = pipe.text_encoder(input_ids)[0].reshape(1,-1)
                embedding_dict.append(cls_emb/cls_emb.norm(dim=-1))
                index_dict[k] = i
        embedding_dict = torch.cat(embedding_dict, dim=0)

    print(f"Evaluating...")
    total_results = 0.0
    learned_embs = []
    gt_embs = []
    if args.clip_image_emb:
        for i, batch in enumerate(tqdm.tqdm(test_dataloader)):
            gt_inputs = batch["gt_image"].to("cuda")
            test_inputs = batch["test_image"].to("cuda")
            with torch.no_grad():
                gt_emb = clip_model(pixel_values=gt_inputs, return_dict=True)['pooler_output']
                out_emb = clip_model(pixel_values=test_inputs, return_dict=True)['pooler_output']
                if args.tsne:
                    learned_embs.append(out_emb.reshape(64, -1).cpu())
                    gt_embs.append(gt_emb.reshape(1,-1).cpu())
                score = CosineSim(out_emb.reshape(1, -1), gt_emb.reshape(1,-1))
                results = score.mean()
                total_results += results
        if args.tsne:
            learned_embs = torch.stack(learned_embs, dim=0).numpy()
            np.save("./learned_embs_b4.npy", learned_embs)
            gt_embs = torch.stack(gt_embs, dim=0).numpy()
            np.save("./gt_embs_b4.npy", gt_embs)
            # pca = PCA(n_components=50)
            # pca_result = pca.fit_transform(learned_embs)
            # print(f'PCA reduced shape: {pca_result.shape}')
            # tsne = TSNE(n_components=2, random_state=42)
            # tsne_result = tsne.fit_transform(pca_result)
            # print(f't-SNE reduced shape: {tsne_result.shape}') 
            # plt.figure(figsize=(10, 8))
            # plt.scatter(tsne_result[:, 0], tsne_result[:, 1], cmap='viridis', alpha=0.6)
            # plt.title('t-SNE Visualization after PCA')
            # plt.savefig('./tsne.png')

    elif args.iqa_test:
        for i, batch in enumerate(tqdm.tqdm(test_dataloader)):
            gt_image, test_image = batch["gt_image"].to("cuda"), batch["test_image"].to("cuda")
            score = fid_metric(test_image, gt_image).mean()
            total_results += score
        print(total_results/len(test_dataloader))

    elif args.clip_text_emb:
        for subset in os.listdir(test_path):
            if subset.startswith("_"): continue
            results = 0.0
            gt_failure = 0
            test_failure = 0
            with torch.no_grad():
                for cls_name in os.listdir(f'{test_path}/{subset}'):
                    learned_token_path = os.path.join(test_path, subset, cls_name, "learned_embeds-steps-1000.safetensors")
                    gt_token_path = os.path.join(gt_path, subset, cls_name, "learned_embeds-steps-1000.safetensors")
                    if not os.path.isfile(learned_token_path):
                        test_failure += 1
                        continue
                    if not os.path.isfile(gt_token_path):
                        gt_failure += 1
                        continue

                    # Learned text embeddings
                    pipe.load_textual_inversion(learned_token_path)
                    input_ids = pipe.tokenizer(
                        [prompt], max_length=pipe.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
                    ).input_ids.to("cuda")
                    learned_txt_emb = pipe.text_encoder(input_ids)[0].reshape(1,-1)
                    pipe.unload_textual_inversion()

                    # Ground truth text embeddings
                    pipe.load_textual_inversion(gt_token_path)
                    input_ids = pipe.tokenizer(
                        [prompt], max_length=pipe.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
                    ).input_ids.to("cuda")
                    gt_txt_emb = pipe.text_encoder(input_ids)[0].reshape(1,-1)
                    pipe.unload_textual_inversion()

                    if args.imagenet_cls:
                        cls_string = f"a photo of a {IMAGENET2012_CLASSES[cls_name]}"
                        input_ids = pipe.tokenizer(
                            [cls_string], max_length=pipe.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
                        ).input_ids.to("cuda")
                        cls_emb = pipe.text_encoder(input_ids)[0].reshape(1,-1)
                        cos_sim_bk = CosineSim(learned_txt_emb, cls_emb)
                        cos_sim_gt = CosineSim(gt_txt_emb, cls_emb)
                        if cos_sim_bk.item() >= cos_sim_gt.item():
                            results += 1.0
                    else:
                        cos_sim = CosineSim(learned_txt_emb, gt_txt_emb)
                        results += cos_sim.item()

                    # # Categorical KLD
                    # learned_txt_emb_score = torch.nn.functional.log_softmax(torch.matmul(learned_txt_emb/learned_txt_emb.norm(dim=-1), embedding_dict.T), dim=1)
                    # gt_txt_emb_score = torch.nn.functional.log_softmax(torch.matmul(gt_txt_emb/gt_txt_emb.norm(dim=-1), embedding_dict.T), dim=1)
                    # cls_kld = kld(learned_txt_emb_score, gt_txt_emb_score)
                    # results += cls_kld.item()
                    # cls_ind = cls_score.reshape(-1).topk(10, dim=-1).indices.tolist()
                    # # Top-K CLIP classification
                    # if index_dict[gt_cls] in cls_ind:
                    #     results += 1
                print(f"{subset}:", results/(len(os.listdir(f'{gt_path}/{subset}'))-test_failure-gt_failure), test_failure, gt_failure)
                total_results += results

    print(total_results/64000)
    print(f"Done: {args.test_path}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', required=True, type=str)
    parser.add_argument('--gt_path', required=True, type=str)
    parser.add_argument('--clip_path', type=str)
    parser.add_argument('--sd_path', type=str)
    parser.add_argument('--iqa_test', action='store_true')
    parser.add_argument('--clip_image_emb', action='store_true')
    parser.add_argument('--clip_text_emb', action='store_true')
    parser.add_argument('--imagenet_cls', action='store_true')
    parser.add_argument('--tsne', action='store_true')
    args = parser.parse_args()
    main(args)
