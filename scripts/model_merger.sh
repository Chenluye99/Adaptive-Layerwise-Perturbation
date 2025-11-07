HF_MODEL_NAME=Chenlu123/Qwen2.5-7B_ppo_is_cliph2.0_clipl0.5_clipc5.0_step100
HF_REPO_ID=$HF_MODEL_NAME  # HuggingFace仓库名称
# HF_MODEL_PATH用于加载config，使用模型名称
HF_MODEL_PATH=Qwen/Qwen2.5-7B
# HF_MODEL_DIR是本地模型路径，用于复制tokenizer文件（如果存在）
HF_MODEL_DIR=/home/chenluy/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B/snapshots/d149729398750b98c0af14eb82c78cfe92750796
CHECKPOINT_DIR=/opt/dlami/nvme/TIR/simpletir_cliph2.0_clipl0.5_clipc5.0_maxturn5_maskvoidTrue_oversample1_losscum-token_ppogeoFalse_risTrue_isth5.0_islvlsequence_maxpro16000_maxres8000_batch128_ppomini32_simplelr_math_35_train_deepscaler_train_Qwen2.5-7B
STEP=100
TARGET_DIR=/opt/dlami/nvme/TIR/output_models/Qwen2.5-7B_ppo_is_cliph2.0_clipl0.5_clipc5.0_step100  # 改为本地路径
# 是否上传到HuggingFace（设置为true则上传）
UPLOAD_TO_HF=true

# 创建输出目录
mkdir -p $TARGET_DIR

python model_merger.py \
    --backend fsdp \
    --hf_model_path $HF_MODEL_PATH \
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

# 上传到HuggingFace
if [ "$UPLOAD_TO_HF" = true ]; then
    echo "Uploading model to HuggingFace..."
    hf upload $HF_MODEL_NAME $TARGET_DIR --repo-type model
    echo "Model uploaded to HuggingFace."
fi