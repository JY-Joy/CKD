#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import logging
import math
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import shutil
import warnings
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import PIL
import safetensors
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from torch.autograd import grad as torch_grad
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder

from packaging import version
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

import diffusers
from datasets import load_dataset
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionXLPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.attention_processor import AttnProcessor


if is_wandb_available():
    import wandb

if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        "linear": PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }

if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("10.0.0"):
    PIL.Image.ANTIALIAS=PIL.Image.LANCZOS
# ------------------------------------------------------------------------------


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.32.0.dev0")

logger = get_logger(__name__)


def binary_mask_eval(args, model: str, pretrained_model_path="", pruned_model_path="", dtype=torch.float16):

    model = model.lower()
    # load sdxl model
    pruned_pipe = StableDiffusionXLPipeline.from_pretrained(
        pretrained_model_path, torch_dtype=dtype
    ).to("cpu")
    pruned_pipe.unet = torch.load(
        pruned_model_path,
        map_location="cpu",
    )

    # FLUX
    # elif model == "flux":
    #     pruned_pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16).to(
    #         "cpu"
    #     )
    #     pruned_pipe.transformer = torch.load(
    #         hf_hub_download("zhangyang-0123/EcoDiffPrunedModels", "model/flux/flux.pkl"),
    #         map_location="cpu",
    #     )

    # reload the original model
    # elif model == "flux":
    #     pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16).to("cpu")

    print("prune model loaded")
    return pruned_pipe


def log_validation(text_encoder, tokenizer, unet, vae, args, accelerator, weight_dtype, step):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
        f" {args.validation_prompt}."
    )
    # create pipeline (note: unet and vae are loaded again in float32)
    pipeline = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        text_encoder=accelerator.unwrap_model(text_encoder),
        tokenizer=tokenizer,
        unet=accelerator.unwrap_model(unet),
        vae=vae,
        safety_checker=None,
        torch_dtype=weight_dtype,
    )
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    generator = None if args.seed is None else torch.Generator(device=accelerator.device).manual_seed(args.seed)
    images = []
    for _ in range(args.num_validation_images):
        autocast_ctx = torch.autocast(accelerator.device.type)

        with autocast_ctx:
            image = pipeline(args.validation_prompt, num_inference_steps=25, generator=generator).images[0]
        images.append(image)

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images("validation", np_images, step, dataformats="NHWC")
        if tracker.name == "wandb":
            tracker.log(
                {
                    "validation": [
                        wandb.Image(image, caption=f"{i}: {args.validation_prompt}") for i, image in enumerate(images)
                    ]
                }
            )

    del pipeline
    torch.cuda.empty_cache()
    return images


def save_progress(text_encoder, placeholder_token_ids, accelerator, args, save_path, safe_serialization=True):
    logger.info("Saving embeddings")
    learned_embeds = (
        accelerator.unwrap_model(text_encoder)
        .get_input_embeddings()
        .weight[min(placeholder_token_ids) : max(placeholder_token_ids) + 1]
    )
    learned_embeds_dict = {args.placeholder_token: learned_embeds.detach().cpu()}

    if safe_serialization:
        safetensors.torch.save_file(learned_embeds_dict, save_path, metadata={"format": "pt"})
    else:
        torch.save(learned_embeds_dict, save_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--num_tokens",
        type=int,
        default=1,
        help="How many textual inversion tokens to be inserted in caption.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument("--unet_config_path", type=str, default="./src/unet_config_v2-base/bk_small.yaml")
    parser.add_argument(
        "--pruned_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--placeholder_token",
        type=str,
        default=None,
        help="A token to use as a placeholder for the concept.",
    )
    parser.add_argument(
        "--initializer_token", type=str, default=None, help="A token to use as initializer word."
    )
    parser.add_argument(
        "--train_data_dir", type=str, default=None, help="A folder containing the training data."
    )
    parser.add_argument("--learnable_property", type=str, default="object", help="Choose between 'object' and 'style'")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--train_size",
        type=int,
        default=1024,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--interpolate_text",
        type=float,
        default=0.0,
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--gm_batch",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and Nvidia Ampere GPU or Intel Gen 4 Xeon (and later) ."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=None,
        help=(
            "Deprecated in favor of validation_steps. Run validation every X epochs. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--no_safe_serialization",
        action="store_true",
        help="If specified save the checkpoint not in `safetensors` format, but in original PyTorch format instead.",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


imagenet_templates_small = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
]

imagenet_style_templates_small = [
    "a painting in the style of {}",
    "a rendering in the style of {}",
    "a cropped painting in the style of {}",
    "the painting in the style of {}",
    "a clean painting in the style of {}",
    "a dirty painting in the style of {}",
    "a dark painting in the style of {}",
    "a picture in the style of {}",
    "a cool painting in the style of {}",
    "a close-up painting in the style of {}",
    "a bright painting in the style of {}",
    "a cropped painting in the style of {}",
    "a good painting in the style of {}",
    "a close-up painting in the style of {}",
    "a rendition in the style of {}",
    "a nice painting in the style of {}",
    "a small painting in the style of {}",
    "a weird painting in the style of {}",
    "a large painting in the style of {}",
]


def main():
    args = parse_args()
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load tokenizer
    tokenizer_1 = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_2")

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder_1 = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
    )
    
    bk_unet = torch.load(args.pruned_model_name_or_path, map_location="cpu", weights_only=False)

    attn_procs = {}
    for k,v in unet.attn_processors.items():
        attn_procs[k] = AttnProcessor()
    unet.set_attn_processor(attn_procs)
    attn_procs = {}
    for k,v in bk_unet.attn_processors.items():
        attn_procs[k] = AttnProcessor()
    bk_unet.set_attn_processor(attn_procs)

    unet_param_count = 0
    for param in unet.parameters():
        unet_param_count += param.numel()
    bk_unet_param_count = 0
    for param in bk_unet.parameters():
        bk_unet_param_count += param.numel()
    logger.info(f"Unet param count: {unet_param_count / 1024 / 1024 / 1024} B")
    logger.info(f"BK Unet param count: {bk_unet_param_count / 1024 / 1024 / 1024} B")

    if args.interpolate_text > 0 and args.placeholder_token is not None:
        # Add the placeholder token in tokenizer
        placeholder_tokens = [args.placeholder_token]

        # if args.num_tokens < 1:
        #     raise ValueError(f"--num_tokens has to be larger or equal to 1, but is {args.num_tokens}")
        #     # add dummy tokens for multi-vector
        #     additional_tokens = []
        #     for i in range(1, args.train_batch_size):
        #         additional_tokens.append(f"{args.placeholder_token}_{i}")
        #     placeholder_tokens += additional_tokens
        # num_added_tokens = tokenizer.add_tokens(placeholder_tokens)
        # if num_added_tokens != args.train_batch_size:
        #     raise ValueError(
        #         f"The tokenizer already contains the token {args.placeholder_token}. Please pass a different"
        #         " `placeholder_token` that is not already in the tokenizer."
        #     )

        # Convert the initializer_token, placeholder_token to ids
        # token_ids = tokenizer.encode(args.initializer_token, add_special_tokens=False)
        # Check if initializer_token is a single token or a sequence of tokens
        # if len(token_ids) > 1:
        #     raise ValueError("The initializer token must be a single token.")

        # initializer_token_id = token_ids[0]
        placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)

        # Resize the token embeddings as we are adding new special tokens to the tokenizer
        text_encoder.resize_token_embeddings(len(tokenizer))

        # # Initialise the newly added placeholder token with the embeddings of the initializer token
        token_embeds = text_encoder.get_input_embeddings().weight.data
        # with torch.no_grad():
        #     for token_id in placeholder_token_ids:
        #         token_embeds[token_id] = token_embeds[initializer_token_id].clone()

    # Freeze vae and unet
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder_1.requires_grad_(False)
    text_encoder_2.requires_grad_(False)

    # FIXME: deprecated? gradient matching on token_embedding level
    # text_encoder.text_model.embeddings.token_embedding.requires_grad_(True)

    if args.gradient_checkpointing:
        # Keep unet in train mode if we are using gradient checkpointing to save memory.
        # The dropout cannot be != 0 so it doesn't matter if we are in eval or train mode.
        # text_encoder.gradient_checkpointing_enable()
        unet.enable_gradient_checkpointing()
        bk_unet.enable_gradient_checkpointing()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
            bk_unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Loss function
    loss_fn = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

    # Initialize the optimizer
    params_to_optimize = bk_unet.parameters()
    optimizer_unet = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Dataset and DataLoaders creation:
    if args.debug:
        from datasets import Dataset
        def dummy_data_generator():
            for i in range(100):
                yield {
                    "res": args.train_size,
                    "caption": "",
                }
        train_dataset = Dataset.from_generator(dummy_data_generator)
    else:
        train_dataset = load_dataset(
            args.train_data_dir,
            data_dir=args.train_data_dir,
            cache_dir=f"{args.train_data_dir}/.cache",
            num_proc=8,
            split="train",
            trust_remote_code=True
        )

    # Preprocessing the datasets.
    column_names = train_dataset.column_names
    image_column = column_names[0]
    caption_column = column_names[1]

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        input_ids_1 = tokenizer_1(
            captions,
            padding="max_length",
            truncation=True,
            max_length=tokenizer_1.model_max_length,
            return_tensors="pt"
        ).input_ids
        
        input_ids_2 = tokenizer_2(
            captions,
            padding="max_length",
            truncation=True,
            max_length=tokenizer_2.model_max_length,
            return_tensors="pt"
        ).input_ids
        return input_ids_1, input_ids_2

    # Preprocessing the datasets.
    train_transforms = transforms.Compose(
        [
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    train_crop = transforms.CenterCrop(args.train_size) if args.center_crop else transforms.RandomCrop(args.train_size)

    def preprocess_train(examples):
        if args.debug:
            examples["pixel_values"] = [torch.randn(3,res,res) for res in examples[image_column]]
            examples["original_size"] = [(1024, 1024) for _ in examples[image_column]]
            examples["crop_top_left"] = [(0, 0) for _ in examples[image_column]]
        else:
            images = []
            original_size = []
            crop_top_left = []
            for image in examples[image_column]:
                if isinstance(image, PIL.Image.Image):
                    if not image.mode == "RGB":
                        image = image.convert("RGB")
                elif isinstance(image, str):
                    image = Image.open(image)
                    if not image.mode == "RGB":
                        image = image.convert("RGB")
                else:
                    raise ValueError(
                        f"Image column `{image_column}` should contain either PIL images or lists of images."
                    )
                original_size.append((image.height, image.width))
                image = image.resize((args.train_size, args.train_size), resample=args.interpolation)
                if args.center_crop:
                    y1 = max(0, int(round((image.height - args.train_size) / 2.0)))
                    x1 = max(0, int(round((image.width - args.train_size) / 2.0)))
                    image = train_crop(image)
                else:
                    y1, x1, h, w = train_crop.get_params(image, (args.train_size, args.train_size))
                    image = transforms.functional.crop(image, y1, x1, h, w)

                crop_top_left.append((y1, x1))
                image = train_transforms(image)
                images.append(image)
            examples["pixel_value"] = images
            examples["original_size"] = original_size
            examples["crop_top_left"] = crop_top_left
        input_ids = tokenize_captions(examples)
        examples["input_ids_1"] = input_ids[0]
        examples["input_ids_2"] = input_ids[1]
        return examples

    with accelerator.main_process_first():
        # Set the training transforms
        train_dataset = train_dataset.with_transform(preprocess_train)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids_1 = torch.stack([example["input_ids_1"] for example in examples])
        input_ids_1 = input_ids_1.to(memory_format=torch.contiguous_format).long()
        input_ids_2 = torch.stack([example["input_ids_2"] for example in examples])
        input_ids_2 = input_ids_2.to(memory_format=torch.contiguous_format).long()
        original_size = [example["original_size"] for example in examples]
        crop_top_left = [example["crop_top_left"] for example in examples]
        return {
            "pixel_values": pixel_values,
            "input_ids_1": input_ids_1,
            "input_ids_2": input_ids_2,
            "original_size": original_size,
            "crop_top_left": crop_top_left,
        }

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    if args.validation_epochs is not None:
        warnings.warn(
            f"FutureWarning: You are doing logging with validation_epochs={args.validation_epochs}."
            " Deprecated validation_epochs in favor of `validation_steps`"
            f"Setting `args.validation_steps` to {args.validation_epochs * len(train_dataset)}",
            FutureWarning,
            stacklevel=2,
        )
        args.validation_steps = args.validation_epochs * len(train_dataset)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer_unet,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
    )

    bk_unet.train()
    # Prepare everything with our `accelerator`.
    bk_unet, optimizer_unet, train_dataloader, lr_scheduler = accelerator.prepare(
        bk_unet, optimizer_unet, train_dataloader, lr_scheduler
    )

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for model in models:
                if isinstance(model, UNet2DConditionModel):
                    torch.save(model.state_dict(), os.path.join(output_dir, "model_ckpt.pt"))
                weights.pop()

    def load_model_hook(models, input_dir):

        while len(models) > 0:
            # pop models so that they are not loaded again
            model = models.pop()

            if isinstance(model, UNet2DConditionModel):
                aggregator_state_dict = torch.load(os.path.join(input_dir, "model_ckpt.pt"), map_location="cpu")
                model.load_state_dict(aggregator_state_dict)

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae and unet to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder_1.to(accelerator.device, dtype=weight_dtype)
    text_encoder_2.to(accelerator.device, dtype=weight_dtype)
    bk_unet.to(accelerator.device)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("textual_inversion", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )


    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(bk_unet):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample().detach()
                latents = latents * vae.config.scaling_factor

                # Get the text embedding for conditioning
                with torch.no_grad():
                    # Get the text embedding for conditioning
                    encoder_hidden_states_1 = (
                        text_encoder_1(batch["input_ids_1"], output_hidden_states=True)
                        .hidden_states[-2]
                        .to(dtype=weight_dtype)
                    )
                    encoder_output_2 = text_encoder_2(batch["input_ids_2"], output_hidden_states=True)
                    encoder_hidden_states_2 = encoder_output_2.hidden_states[-2].to(dtype=weight_dtype)
                    original_size = [
                        (batch["original_size"][i][0].item(), batch["original_size"][i][1].item())
                        for i in range(args.train_batch_size)
                    ]
                    crop_top_left = [
                        (batch["crop_top_left"][0][i].item(), batch["crop_top_left"][1][i].item())
                        for i in range(args.train_batch_size)
                    ]
                    target_size = (args.resolution, args.resolution)
                    add_time_ids = torch.cat(
                        [
                            torch.tensor(original_size[i] + crop_top_left[i] + target_size)
                            for i in range(args.train_batch_size)
                        ]
                    ).to(accelerator.device, dtype=weight_dtype)
                    print(add_time_ids.shape)
                    exit()
                    added_cond_kwargs = {"text_embeds": encoder_output_2[0], "time_ids": add_time_ids}
                    encoder_hidden_states = torch.cat([encoder_hidden_states_1, encoder_hidden_states_2], dim=-1)
                    if args.interpolate_text > 0:
                        invert_captions = []
                        for ph_token_id, ph_token in zip(placeholder_token_ids, placeholder_tokens):
                            token_embeds[ph_token_id] = torch.randn_like(token_embeds[ph_token_id])
                            inversion_text = random.choice(imagenet_templates_small).format(ph_token)
                            invert_captions.append(inversion_text)
                        inversion_text_ids = tokenizer(
                            invert_captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
                        ).input_ids
                        inversion_text_ids = inversion_text_ids.to(accelerator.device)
                        inversion_hidden_states = text_encoder(inversion_text_ids)[0].to(dtype=weight_dtype)
                        # random interpolate two hidden_states
                        # inversion_hidden_states = torch.randn_like(encoder_hidden_states)
                        gamma = torch.rand(encoder_hidden_states.shape[0], 1, 1, device=accelerator.device) * args.interpolate_text
                        noisy_encoder_hidden_states = (encoder_hidden_states - inversion_hidden_states) * gamma + inversion_hidden_states
                        noisy_encoder_hidden_states = noisy_encoder_hidden_states.to(weight_dtype)
                        noisy_encoder_hidden_states.requires_grad = True
                    else:
                        noisy_encoder_hidden_states = encoder_hidden_states.to(dtype=weight_dtype)
                        noisy_encoder_hidden_states.requires_grad = True

                # Gradient Matching
                for _ in range(args.gm_batch):

                    # Forward diffusion process
                    # 1. Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    # 2. Sample a random timestep for each image
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                    timesteps = timesteps.long()

                    # 3. Add noise to the latents according to the noise magnitude at each timestep
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    # 4. Get the target for loss depending on the prediction type
                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                    # # Diffusion Loss
                    teacher_pred = unet(noisy_latents, timesteps, noisy_encoder_hidden_states).sample
                    teacher_loss = F.mse_loss(teacher_pred.float(), target.float(), reduction="mean")

                    # Gradient Matching
                    target_gradient = torch_grad(outputs=teacher_loss, inputs=noisy_encoder_hidden_states)[0]
                    student_pred = bk_unet(noisy_latents, timesteps, noisy_encoder_hidden_states).sample
                    student_loss = F.mse_loss(student_pred.float(), target.float(), reduction="mean")
                    student_gradient = torch_grad(outputs=student_loss, inputs=noisy_encoder_hidden_states,
                                                create_graph=True, retain_graph=True, only_inputs=True
                                        )[0]

                    # mse similarity
                    sim = F.mse_loss(student_gradient.float(), target_gradient.float(), reduction="mean")
                    loss_gm = sim / args.gm_batch

                    # cosine similarity
                    # sim = loss_fn(student_gradient.float(), target_gradient.float())
                    # loss_gm = (1 - sim).mean() / args.gm_batch
                    # accelerator.backward(loss_gm)  # maximize the similarity

                # Diffusion Loss
                # student_pred = bk_unet(noisy_latents, timesteps, encoder_hidden_states).sample
                loss_distill = F.mse_loss(student_pred.float(), teacher_pred.detach().float(), reduction="mean")
                loss = 1000 * loss_gm + loss_distill + student_loss
                accelerator.backward(loss)
                # total_grad_norm = accelerator.clip_grad_norm_(params_to_optimize, 1.0)
                # optimizer_unet.step()
                # lr_scheduler.step()
                # optimizer_unet.zero_grad()
                # text_encoder.text_model.embeddings.token_embedding.grad = None

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                images = []
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    if args.validation_prompt is not None and global_step % args.validation_steps == 0:
                        images = log_validation(
                            text_encoder, tokenizer, bk_unet, vae, args, accelerator, weight_dtype, global_step
                        )

            logs = {
                "grad_sim": sim.detach().item(),
                "stu_diff": student_loss.detach().item(),
                "stu_grad": student_gradient.detach().norm().item(),
                "tar_grad": target_gradient.detach().norm().item(),
                "lr": lr_scheduler.get_last_lr()[0]
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
