from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="black-forest-labs/FLUX.1-Fill-dev",
    local_dir="models/flux-fill",
    local_dir_use_symlinks=False  # symlink 대신 실제 파일 복사
)