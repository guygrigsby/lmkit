# Cloud GPU job image: lmkit plus the Gemma 4 LoRA stack, entrypoint = lmkit.cloudjob.
# Base is the image the throughput numbers in examples/gemma_cpt were measured on
# (torch 2.8, CUDA 12.8, Python 3.12). Changing a pin here changes throughput: re-measure.
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

LABEL org.opencontainers.image.source="https://github.com/guygrigsby/lmkit"

RUN pip install --no-cache-dir --break-system-packages \
      "transformers==5.23.0" "peft==0.22.0" "huggingface_hub[hf_xet]" \
      "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post2/flash_attn-2.8.3.post2+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"

WORKDIR /opt/lmkit
COPY pyproject.toml README.md ./
COPY lmkit ./lmkit
COPY examples ./examples
RUN pip install --no-cache-dir --break-system-packages --no-deps .

ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    HF_HOME=/cache/hf \
    LMKIT_OUT_DIR=/out

# A job is configured entirely by LMKIT_* env vars; see lmkit/cloudjob.py.
ENTRYPOINT ["python3", "-m", "lmkit.cloudjob"]
