# Set Python path
export PYTHONPATH="$(pwd)/third_party/Megatron-LM:${PYTHONPATH}"

# Verify the import works
python3 -c "from megatron.core.datasets import indexed_dataset; print('Import successful')"

# Then run your preparation script
python examples/megatron/prepare_fineweb_edu.py \
    --primus-path . \
    --data-path ./data \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model meta-llama/Llama-3.2-1B \
    --sample-size 10BT