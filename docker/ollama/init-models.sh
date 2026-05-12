#!/bin/sh
set -eu

echo "Waiting for Ollama at ${OLLAMA_HOST:-http://ollama:11434}..."
until ollama list >/dev/null 2>&1; do
  sleep 2
done

echo "Pulling compact public models used as Docker defaults..."
ollama pull qwen2.5:0.5b
ollama pull llama3.2:1b

echo "Creating project aliases for tested model names where possible..."
cat >/tmp/Modelfile.qwen05 <<'EOF'
FROM qwen2.5:0.5b
PARAMETER temperature 0.7
EOF
ollama create qwen2.5-0.5b-instruct -f /tmp/Modelfile.qwen05 || true

cat >/tmp/Modelfile.llama1b <<'EOF'
FROM llama3.2:1b
PARAMETER temperature 0.7
EOF
ollama create llama3.2-1b-instruct -f /tmp/Modelfile.llama1b || true

if [ -f /bootstrap/create-lfm-aliases.sh ]; then
  /bin/sh /bootstrap/create-lfm-aliases.sh || true
fi

echo "Available Ollama models:"
ollama list
