# Zolna backend: Gradio + llama.cpp + hybrid retriever (space/app.py).
# Build context = repository root (see docker-compose.yml). Works on aarch64 (Oracle A1) and x86_64.
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu \
    HF_HOME=/root/.cache/huggingface

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential cmake git curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# llama.cpp is compiled for the CPU of the machine that builds the image (ARM NEON on Oracle A1).
ARG LLAMA_CPP_PYTHON=0.3.33
RUN CMAKE_ARGS="-DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF" \
    CMAKE_BUILD_PARALLEL_LEVEL=4 \
    pip install "llama-cpp-python==${LLAMA_CPP_PYTHON}"

# Remaining dependencies (the requirements' wheel-URL line is only for the x86 Hugging Face Space).
COPY space/requirements.txt /tmp/requirements.txt
RUN grep -v '__WHEEL_URL__' /tmp/requirements.txt > /tmp/req.txt && pip install -r /tmp/req.txt

COPY space/ /app/

EXPOSE 7860
# First start downloads the model + embedder from Hugging Face, hence the long start period.
HEALTHCHECK --interval=60s --timeout=10s --start-period=1200s --retries=3 \
  CMD curl -fs http://127.0.0.1:7860/config > /dev/null || exit 1

CMD ["python", "app.py"]
