HF_MODEL_NAME=Chenlu123/rollout_is_model_global_step_100
HF_REPO_ID=$HF_MODEL_NAME  # HuggingFace仓库名称
# HF_MODEL_DIR应该是本地模型路径，用于获取config和tokenizer
HF_MODEL_DIR=/home/chenluy/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B/snapshots/d149729398750b98c0af14eb82c78cfe92750796
CHECKPOINT_DIR=/opt/dlami/nvme/TIR/simpletir_maxres8000_maxpro16000_maxturn5_batch128_ppomini32_lossmodevanilla_maskvoidturnsTrue_risTrue_islvlsequence_isth5.0_simplelr_math_35_train_deepscaler_train_Qwen2.5-7B
STEP=100
TARGET_DIR=./output_models/rollout_is_model_global_step_100  # 改为本地路径
# 是否上传到HuggingFace（设置为true则上传）
UPLOAD_TO_HF=true

# 创建输出目录
mkdir -p $TARGET_DIR

python scripts/model_merger.py \
    --backend fsdp \
    --hf_model_path $HF_MODEL_DIR \
    --local_dir $CHECKPOINT_DIR/global_step_$STEP/actor \
    --target_dir $TARGET_DIR

# 复制tokenizer相关文件
echo "Copying tokenizer files from $HF_MODEL_DIR to $TARGET_DIR..."
cp $HF_MODEL_DIR/tokenizer.json $TARGET_DIR/ 2>/dev/null || echo "Warning: tokenizer.json not found"
cp $HF_MODEL_DIR/tokenizer_config.json $TARGET_DIR/ 2>/dev/null || echo "Warning: tokenizer_config.json not found"  
cp $HF_MODEL_DIR/vocab.json $TARGET_DIR/ 2>/dev/null || echo "Warning: vocab.json not found"
cp $HF_MODEL_DIR/merges.txt $TARGET_DIR/ 2>/dev/null || echo "Warning: merges.txt not found"
cp $HF_MODEL_DIR/config.json $TARGET_DIR/ 2>/dev/null || echo "Warning: config.json not found"
cp $HF_MODEL_DIR/generation_config.json $TARGET_DIR/ 2>/dev/null || echo "Warning: generation_config.json not found"
echo "Tokenizer files copy completed."
