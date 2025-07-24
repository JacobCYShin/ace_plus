import base64
import os
import uuid
import subprocess
import runpod
from huggingface_hub import snapshot_download

# 🔽 모델 다운로드 (런타임에 환경변수 통해 토큰 사용)
def download_model():
    hf_token = os.getenv("HF_TOKEN")
    snapshot_download(
        repo_id="black-forest-labs/FLUX.1-Fill-dev",
        local_dir="models/flux-fill",
        local_dir_use_symlinks=False,
        token=hf_token,
        ignore_patterns=["*.safetensors"]
    )

# 🔽 요청 처리 함수
def handler(event):
    instruction = event["input"]["instruction"]
    input_image_path = event["input"]["input_reference_image"]
    output_image_path = f"/tmp/{uuid.uuid4().hex}.jpg"

    command = [
        "python3", "infer_lora.py",
        "--instruction", instruction,
        "--input_reference_image", input_image_path,
        "--task_type", "portrait",
        "--task_model", "models/model_zoo.yaml",
        "--cfg_folder", "config",
        "--save_path", output_image_path,
        "--infer_type", "diffusers"
    ]

    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        return {
            "error": "Inference failed",
            "stderr": result.stderr,
            "stdout": result.stdout
        }

    with open(output_image_path, "rb") as img_file:
        image_bytes = img_file.read()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    return {
        "output_image": image_base64
    }

download_model()  # 서버 시작 시 한 번만 다운로드

# ✅ RunPod 서버리스 시작 지점
runpod.serverless.start({"handler": handler})
