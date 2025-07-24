import os
from huggingface_hub import snapshot_download

hf_token = os.getenv("HF_TOKEN")  # RunPod Secret ENV로부터
snapshot_download(
    repo_id="black-forest-labs/FLUX.1-Fill-dev",
    local_dir="models/flux-fill",
    local_dir_use_symlinks=False,
    token=hf_token
)
