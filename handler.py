import base64
import os
import uuid
import subprocess
import runpod


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
