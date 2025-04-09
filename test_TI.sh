# CLS_NAME=robin
# echo $MODEL_ID
# echo $TEST_DIR/$SUB_FOLDER/$CLS_NAME
accelerate launch src/textual_inversion_sdxl.py \
  --pretrained_model_name_or_path /home/jenyuan/zoo/stable-diffusion-xl-base-1.0 \
  --train_data_dir data/ip/starry_night \
  --mixed_precision bf16 \
  --resolution=1024 \
  --learning_rate=5.0e-04 \
  --seed=3467 \
  --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --learnable_property="object" \
  --placeholder_token="<object>" \
  --initializer_token="object" \
  --train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --max_train_steps 1000 \
  --checkpointing_steps 5000 \
  --output_dir TI_test/starry_night
#   --distilled_ckpt /apdcephfs_cq10/share_916081/jentsehuang/TI_distill/sd_1_4/$MODEL_ID/HS_GM_mse/checkpoint-35000/model_ckpt.pt
# echo "DONE: $MODEL_ID | $SUB_FOLDER/$CLS_NAME"
