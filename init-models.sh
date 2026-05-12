#!/bin/bash

# Wait for the main server to wake up
sleep 5

# Register your LM Studio models into Ollama
# Replace 'phi-3-mini.gguf' with your actual filenames
models=("phi3" "gemma" "qwen" "llama3")

for model in "${models[@]}"; do
    echo "Registering $model..."
    # This creates a 'pointer' so Ollama knows where the file is
    ollama run $model "hi" # This pulls it if it's a standard name
done

echo "All models ready!"