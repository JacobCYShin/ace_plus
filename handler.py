import base64
import os
import uuid
import subprocess
import runpod
from huggingface_hub import snapshot_download


MODEL_DIR = "models/flux-fill"

# 🔍 모델이 없을 경우에만 다운로드
if not os.path.exists(MODEL_DIR) or not os.listdir(MODEL_DIR):
    print("📦 모델 다운로드 중...")

    hf_token = os.getenv("HF_TOKEN")  # RunPod의 환경변수에서 받아옴
    if not hf_token:
        raise ValueError("❌ HF_TOKEN 환경변수가 설정되지 않았습니다.")

    snapshot_download(
        repo_id="black-forest-labs/FLUX.1-Fill-dev",
        local_dir=MODEL_DIR,
        local_dir_use_symlinks=False,
        token=hf_token
    )

    print("✅ 모델 다운로드 완료")
else:
    print("🚫 모델 이미 존재함, 다운로드 생략.")

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

# ✅ RunPod 서버리스 시작 지점
runpod.serverless.start({"handler": handler})
