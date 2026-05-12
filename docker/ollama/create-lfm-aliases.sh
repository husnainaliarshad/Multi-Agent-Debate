#!/bin/sh
set -eu

# Optional helper for local LM Studio GGUF weights.
#
# Docker can only create the exact LFM aliases used in the study if the
# corresponding GGUF files exist on the host and are mounted at /lm-models.
# Set these env vars before running `docker compose --profile models up model-init`
# if your filenames differ:
#
#   LFM_350M_GGUF=/lm-models/path/to/lfm2-350m.gguf
#   LFM_12B_GGUF=/lm-models/path/to/lfm2-1.2b.gguf
#
# If they are not set, this script searches common filename patterns.

find_first() {
  pattern="$1"
  find /lm-models -type f -iname "$pattern" 2>/dev/null | head -n 1
}

create_from_gguf() {
  alias="$1"
  gguf_path="$2"
  if [ -z "$gguf_path" ] || [ ! -f "$gguf_path" ]; then
    echo "Skipping ${alias}: GGUF file not found."
    return 0
  fi

  modelfile="/tmp/Modelfile.${alias}"
  {
    echo "FROM ${gguf_path}"
    echo "PARAMETER temperature 0.7"
  } > "$modelfile"
  echo "Creating ${alias} from ${gguf_path}"
  ollama create "$alias" -f "$modelfile"
}

LFM_350M="${LFM_350M_GGUF:-}"
LFM_12B="${LFM_12B_GGUF:-}"

if [ -z "$LFM_350M" ]; then
  LFM_350M="$(find_first '*lfm*350*m*.gguf' || true)"
fi
if [ -z "$LFM_12B" ]; then
  LFM_12B="$(find_first '*lfm*1.2*b*.gguf' || true)"
fi

create_from_gguf "lfm2-350m" "$LFM_350M"
create_from_gguf "lfm2-1.2b" "$LFM_12B"
