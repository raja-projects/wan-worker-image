# Public base for the Wan Studio RunPod workers: open-source Python deps + RIFE weights only.
# No worker code and no secrets: the private handler code arrives at start from the RunPod template env.
FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1
COPY requirements.txt /tmp/requirements.txt
RUN pip install -q uv && uv pip install --system -q -r /tmp/requirements.txt && rm -rf /root/.cache/uv /root/.cache/pip
RUN python -c "import torch; from ccvfi import AutoModel, ConfigType; AutoModel.from_pretrained(ConfigType.RIFE_IFNet_v426_heavy, device=torch.device('cpu'), fp16=False)" || echo "RIFE prefetch skipped"
ENV HF_HUB_ENABLE_HF_TRANSFER=1
