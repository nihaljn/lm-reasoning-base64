# API stuff
model_name="gpt-4o"
temperature="0.0"
max_tokens="1024"
num_threads="25"
rpm_limit="100"

# variants
system_prompt="assistant"
evaluator="english_evaluator"
few_shot_k="0"
num_digits="$1"
num_examples="10000" # big number indicates to use all examples

# paths
data_root="data/arithmetic/"
out_root_dir="model_outputs/"
data_path="${data_root}/addition_${num_digits}digits.csv"
few_shot_data_path="${data_root}/addition_${num_digits}digits_few_shot.csv"
out_root_dir="${out_root_dir}/arithmetic/addition_${num_digits}digits/"

# run
python run_arithmetic_evaluation.py \
    --data_path $data_path \
    --out_root_dir $out_root_dir \
    --few_shot_data_path $few_shot_data_path \
    --temperature $temperature \
    --max_tokens $max_tokens \
    --num_threads $num_threads \
    --rpm_limit $rpm_limit \
    --num_examples $num_examples \
    --system_prompt $system_prompt \
    --evaluator $evaluator \
    --model_name $model_name \
    --few_shot_k $few_shot_k
