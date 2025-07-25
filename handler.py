import base64
import os
import uuid
import subprocess
import runpod
import tempfile
import base64

def handler(event):
    instruction = event["input"]["instruction"]
    encoded_image = event["input"]["input_reference_image"]

    # 1. 임시 파일로 저장
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    with open(temp_file.name, "wb") as f:
        f.write(base64.b64decode(encoded_image))

    output_image_path = f"/tmp/{uuid.uuid4().hex}.jpg"

    command = [
        "python3", "infer_lora.py",
        "--instruction", instruction,
        "--input_reference_image", temp_file.name,   # ✅ 파일 경로만 전달
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

    # ✅ 이미지 파일 삭제 (안전)
    try:
        os.remove(output_image_path)
    except Exception as e:
        print(f"⚠️ 이미지 삭제 실패: {e}")

    return {
        "output_image": image_base64
    }

# ✅ RunPod 서버리스 시작 지점
runpod.serverless.start({"handler": handler})
