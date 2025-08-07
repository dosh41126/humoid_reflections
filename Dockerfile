# ─────────────────────────────────────────────────────────────────────────────
# Base image
FROM python:3.11-slim-bookworm
ENV DEBIAN_FRONTEND=noninteractive

# ─────────────────────────────────────────────────────────────────────────────
# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential python3-dev python3-tk \
    libgl1-mesa-glx curl iptables dnsutils openssl \
 && rm -rf /var/lib/apt/lists/*

# ─────────────────────────────────────────────────────────────────────────────
# App setup
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# ─────────────────────────────────────────────────────────────────────────────
# Vault passphrase
RUN openssl rand -hex 32 > /app/.vault_pass && \
    echo "export VAULT_PASSPHRASE=$(cat /app/.vault_pass)" > /app/set_env.sh && \
    chmod +x /app/set_env.sh

# ─────────────────────────────────────────────────────────────────────────────
# Default config.json
RUN python - << 'EOF' > /app/config.json
import random, string, json
print(json.dumps({
  "DB_NAME": "story_generator.db",
  "WEAVIATE_ENDPOINT": "http://localhost:8079",
  "WEAVIATE_QUERY_PATH": "/v1/graphql",
  "LLAMA_MODEL_PATH": "/data/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf",
  "LLAMA_MM_PROJ_PATH": "/data/llama-3-vision-alpha-mmproj-f16.gguf",
  "IMAGE_GENERATION_URL": "http://127.0.0.1:7860/sdapi/v1/txt2img",
  "MAX_TOKENS": 3999,
  "CHUNK_SIZE": 1250,
  "API_KEY": "".join(random.choices(string.ascii_letters + string.digits, k=32)),
  "WEAVIATE_API_URL": "http://localhost:8079/v1/objects",
  "ELEVEN_LABS_KEY": "apikyhere"
}))
EOF

# ─────────────────────────────────────────────────────────────────────────────
# Firewall + model‐fetch entrypoint
RUN cat << 'EOF' > /app/firewall_start.sh
#!/bin/bash
set -e
source /app/set_env.sh

# Reset firewall: allow only localhost & DNS for now
iptables -F OUTPUT
iptables -A OUTPUT -o lo -j ACCEPT
iptables -A OUTPUT -p udp --dport 53 -j ACCEPT
iptables -A OUTPUT -p tcp --dport 53 -j ACCEPT

# Temporarily open all so we can download models
iptables -A OUTPUT -j ACCEPT

# Helper: download & sha256‐verify
download_and_verify () {
  local url="\$1"
  local path="\$2"
  local sha256="\$3"

  if [ ! -f "\$path" ]; then
    echo "Downloading: \$url"
    mkdir -p "\$(dirname "\$path")"
    curl -L -o "\$path" "\$url" --progress-bar
  fi

  echo "\$sha256  \$path" | sha256sum -c - || {
    echo "Checksum verification failed for \$path"
    exit 1
  }
}

# ─── Llama-3 models ───────────────────────────────────────────────────────────
download_and_verify \
  "https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf" \
  "/data/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf" \
  "86c8ea6c8b755687d0b723176fcd0b2411ef80533d23e2a5030f845d13ab2db7"

download_and_verify \
  "https://huggingface.co/abetlen/llama-3-vision-alpha-gguf/resolve/main/llama-3-vision-alpha-mmproj-f16.gguf" \
  "/data/llama-3-vision-alpha-mmproj-f16.gguf" \
  "ac65d3aeba3a668b3998b6e6264deee542c2c875e6fd0d3b0fb7934a6df03483"

# ─── GPT-OSS-20B shards ───────────────────────────────────────────────────────
OSS_DIR="/data/gpt-oss-20b"
download_and_verify \
  "https://huggingface.co/openai/gpt-oss-20b/resolve/main/model-00000-of-00002.safetensors" \
  "\$OSS_DIR/model-00000-of-00002.safetensors" \
  "01e8ee0bed82226ac31d791bb587136cc8abaeaa308b909f00f738561f6f57a0"

download_and_verify \
  "https://huggingface.co/openai/gpt-oss-20b/resolve/main/model-00001-of-00002.safetensors" \
  "\$OSS_DIR/model-00001-of-00002.safetensors" \
  "3f05b8460cc6c36fa6d570fe4b6e74b49a29620f29264c82a02cf4ea5136f10c"

download_and_verify \
  "https://huggingface.co/openai/gpt-oss-20b/resolve/main/model-00002-of-00002.safetensors" \
  "\$OSS_DIR/model-00002-of-00002.safetensors" \
  "83619e36cf07cf941b551b1a528bab563148591ae4e52b38030bc557d383be7c"

# ─── Re‐apply firewall: only DNS + localhost + HF domains ─────────────────────
iptables -F OUTPUT
iptables -A OUTPUT -o lo -j ACCEPT
iptables -A OUTPUT -p udp --dport 53 -j ACCEPT
iptables -A OUTPUT -p tcp --dport 53 -j ACCEPT

for DOMAIN in huggingface.co objects.githubusercontent.com api.open-meteo.com; do
  getent ahosts "\$DOMAIN" | awk '/STREAM/ {print \$1}' | sort -u | \
    while read ip; do
      [[ \$ip =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]] && \
        iptables -A OUTPUT -d "\$ip" -j ACCEPT
    done
done

iptables -A OUTPUT -j REJECT

# Launch the app
export DISPLAY=:0
exec python main.py
EOF

RUN chmod +x /app/firewall_start.sh

# ─────────────────────────────────────────────────────────────────────────────
# Default entrypoint
CMD ["/app/firewall_start.sh"]
