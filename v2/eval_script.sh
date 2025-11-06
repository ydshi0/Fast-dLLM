# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
model_path='/shared_LLM_model/dllm/Efficient-Large-Model/Fast_dLLM_v2_7B'
export CUDA_VISIBLE_DEVICES=7

# task=mmlu
# accelerate launch eval.py --tasks ${task} --batch_size 1 --num_fewshot 5 \
# --confirm_run_unsafe_code --model fast_dllm_v2 --fewshot_as_multiturn --apply_chat_template \
# --model_args model_path=${model_path}

# task=gpqa_main_n_shot
# accelerate launch eval.py --tasks ${task} --batch_size 1 \
# --confirm_run_unsafe_code --model fast_dllm_v2 --fewshot_as_multiturn --apply_chat_template \
# --model_args model_path=${model_path}

task=gsm8k
# nsys profile -o report.nsys-rep --trace-fork-before-exec=true --cuda-graph-trace=node --force-overwrite true \
accelerate launch eval.py --tasks ${task} --batch_size 64 --num_fewshot 0 \
--confirm_run_unsafe_code --model fast_dllm_v2 --fewshot_as_multiturn --apply_chat_template \
--model_args model_path=${model_path},threshold=1,show_speed=True \
--limit 384 --seed 42

# task=minerva_math
# accelerate launch eval.py --tasks ${task} --batch_size 32 --num_fewshot 0 \
# --confirm_run_unsafe_code --model fast_dllm_v2 --fewshot_as_multiturn --apply_chat_template \
# --model_args model_path=${model_path},threshold=1,show_speed=True

# task=ifeval
# accelerate launch eval.py --tasks ${task} --batch_size 32 \
# --confirm_run_unsafe_code --model fast_dllm_v2 --fewshot_as_multiturn --apply_chat_template \
# --model_args model_path=${model_path},threshold=1,show_speed=True
