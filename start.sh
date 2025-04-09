export SUB_FOLDER=subset_19

conda activate wespeaker_py3.10_pytorch1.12.1_cuda11.3
export MODEL_ID="small"
# ./textual_inversion.sh

# evaluate
python3 test_metrics.py \
    --iqa_test \
    --test_path /apdcephfs_cq8/share_916081/jentsehuang/TI_distill/imagenet_test_subset/sd_1_4/image_samples/tiny/HS_GM_mse_diff_loss/checkpoint-35000 \
    --gt_path /apdcephfs_cq10/share_916081/jentsehuang/data/imagenet_test \
    --clip_path /apdcephfs/default121133/apdcephfs_qy3/share_301812049/jentsehuang/models/openai/clip-vit-base-patch32 \
    --sd_path /apdcephfs/default121133/apdcephfs_qy3/share_301812049/jentsehuang/models/stable-diffusion-v1-4 \
    > /apdcephfs_cq8/share_916081/jentsehuang/tiny_ckd.txt

# # sample
# export TEST_DIR="/apdcephfs_cq8/share_916081/jentsehuang/TI_distill/imagenet_test_subset/sd_1_4"
# python3 sample_images.py --test_path $TEST_DIR/$MODEL_ID/bk_model/$SUB_FOLDER --out_path $TEST_DIR/image_samples/$MODEL_ID/bk_model

# # Textual inversion
# export TEST_DIR="/apdcephfs_cq10/share_916081/jentsehuang/data/imagenet_test_subset"
# file_array=($(ls $TEST_DIR/$SUB_FOLDER))
# file_num=${#file_array[@]}
# for ((i=40; i<50; i+=1)); do
#     export CLS_NAME=${file_array[i]}
#     ./Tinversion.sh
# done

# CQ_ROOT=/apdcephfs_cq10/share_916081/jentsehuang
# QY_ROOT=/apdcephfs_qy3/share_301812049/jentsehuang
# CLUSTER_ROOT=$CQ_ROOT
# TRIAL_NAME=TI_distill/sd_1_4/$MODEL_ID/HS_GM_mse_diff_loss
# accelerate launch src/textual_inversion_distillation_gm.py \
#   --logging_dir $CLUSTER_ROOT/$TRIAL_NAME/logs \
#   --output_dir $CLUSTER_ROOT/$TRIAL_NAME \
#   --pretrained_model_name_or_path $CLUSTER_ROOT/models/stable-diffusion-v1-4 \
#   --bk_model_name_or_path $CLUSTER_ROOT/models/bk-sdm-$MODEL_ID \
#   --train_data_dir $CLUSTER_ROOT/data/laion_aes/preprocessed_2256k \
#   --resolution=512 \
#   --seed=3467 \
#   --mixed_precision fp16 \
#   --validation_prompt "a photo of a dog wandering in the sky" \
#   --learning_rate 1.0e-5 \
#   --train_batch_size 1 \
#   --gradient_accumulation_steps 8 \
#   --resume_from_checkpoint latest \
#   --max_train_steps 35000 \
#   --checkpointing_steps 5000 \
#   --validation_steps 1000
#   --placeholder_token="<object>" \
#   --initializer_token="object" \
#   --interpolate_text 1.0
#   --unet_config_name bk_$MODEL_ID \
#   --use_copy_weight_from_teacher \
#   --unet_config_path ./src/unet_config_v2-base/bk_$MODEL_ID
