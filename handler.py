# -*- coding: utf-8 -*-
import base64
import os
import uuid
import tempfile
import runpod
import io
import glob
from PIL import Image

# ACE Plus 관련 import
from scepter.modules.transform.io import pillow_convert
from scepter.modules.utils.config import Config
from scepter.modules.utils.file_system import FS
from inference.ace_plus_diffusers import ACEPlusDiffuserInference

# 전역 변수로 모델 파이프라인 저장
pipe = None
task_model_dict = {}
initialization_complete = False

def initialize_model():
    """모델을 미리 로드하는 함수"""
    global pipe, task_model_dict, initialization_complete
    
    print("🚀 Starting model initialization...")
    
    # FS 클라이언트 초기화
    fs_list = [
        Config(cfg_dict={"NAME": "HuggingfaceFs", "TEMP_DIR": "./cache"}, load=False),
        Config(cfg_dict={"NAME": "ModelscopeFs", "TEMP_DIR": "./cache"}, load=False),
        Config(cfg_dict={"NAME": "HttpFs", "TEMP_DIR": "./cache"}, load=False),
        Config(cfg_dict={"NAME": "LocalFs", "TEMP_DIR": "./cache"}, load=False),
    ]
    
    for one_fs in fs_list:
        FS.init_fs_client(one_fs)
    
    # 설정 파일 로드
    cfg_folder = "./config"
    model_yamls = glob.glob(os.path.join(cfg_folder, '*.yaml'))
    model_choices = dict()
    for i in model_yamls:
        model_cfg = Config(load=True, cfg_file=i)
        model_name = model_cfg.NAME
        model_choices[model_name] = model_cfg
    
    # diffusers 인퍼런스 설정 로드
    infer_name = "ace_plus_diffuser_infer"
    assert infer_name in model_choices
    
    # 태스크 모델 설정 로드
    task_model_cfg = Config(load=True, cfg_file="models/model_zoo.yaml")
    for task_name, task_model in task_model_cfg.MODEL.items():
        task_model_dict[task_name] = task_model
    
    # 파이프라인 초기화
    pipe_cfg = model_choices[infer_name]
    pipe = ACEPlusDiffuserInference()
    pipe.init_from_cfg(pipe_cfg)
    
    initialization_complete = True
    print("✅ Model initialization complete!")

def run_inference(pipe,
                 input_image=None,
                 input_mask=None,
                 input_reference_image=None,
                 save_path="examples/output/example.png",
                 instruction="",
                 output_h=1024,
                 output_w=1024,
                 seed=-1,
                 sample_steps=None,
                 guide_scale=None,
                 repainting_scale=None,
                 model_path=None,
                 **kwargs):
    """실제 추론을 수행하는 함수"""
    if input_image is not None:
        input_image = Image.open(io.BytesIO(FS.get_object(input_image)))
        input_image = pillow_convert(input_image, "RGB")
    if input_mask is not None:
        input_mask = Image.open(io.BytesIO(FS.get_object(input_mask)))
        input_mask = pillow_convert(input_mask, "L")
    if input_reference_image is not None:
        input_reference_image = Image.open(io.BytesIO(FS.get_object(input_reference_image)))
        input_reference_image = pillow_convert(input_reference_image, "RGB")

    image, seed = pipe(
        reference_image=input_reference_image,
        edit_image=input_image,
        edit_mask=input_mask,
        prompt=instruction,
        output_height=output_h,
        output_width=output_w,
        sampler='flow_euler',
        sample_steps=sample_steps or pipe.input.get("sample_steps", 28),
        guide_scale=guide_scale or pipe.input.get("guide_scale", 50),
        seed=seed,
        repainting_scale=repainting_scale or pipe.input.get("repainting_scale", 1.0),
        lora_path=model_path
    )
    
    with FS.put_to(save_path) as local_path:
        image.save(local_path)
    return local_path, seed

def handler(event):
    """RunPod 핸들러 함수"""
    global pipe, task_model_dict, initialization_complete
    
    # 초기화가 완료되지 않았으면 에러 반환
    if not initialization_complete:
        return {
            "error": "Model still initializing. Please try again in a few seconds.",
            "status": "initializing"
        }
    
    try:
        # 입력 파라미터 추출
        instruction = event["input"]["instruction"]
        encoded_image = event["input"]["input_reference_image"]
        task_type = event["input"].get("task_type", "portrait")
        output_h = event["input"].get("output_h", 1024)
        output_w = event["input"].get("output_w", 1024)
        seed = event["input"].get("seed", -1)
        
        # 임시 파일로 이미지 저장
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        with open(temp_file.name, "wb") as f:
            f.write(base64.b64decode(encoded_image))
        
        output_image_path = f"/tmp/{uuid.uuid4().hex}.jpg"
        
        # 태스크 타입에 따른 모델 경로 설정
        task_type_upper = task_type.upper()
        if task_type_upper not in task_model_dict:
            return {
                "error": f"Unsupported task type: {task_type}",
                "supported_types": list(task_model_dict.keys())
            }
        
        model_path = FS.get_from(task_model_dict[task_type_upper]["MODEL_PATH"])
        repainting_scale = task_model_dict[task_type_upper].get("REPAINTING_SCALE", 1.0)
        
        # 추론 실행
        params = {
            "input_reference_image": temp_file.name,
            "save_path": output_image_path,
            "instruction": instruction,
            "output_h": output_h,
            "output_w": output_w,
            "seed": seed,
            "repainting_scale": repainting_scale,
            "model_path": model_path
        }
        
        local_path, generated_seed = run_inference(pipe, **params)
        
        # 결과 이미지를 base64로 인코딩
        with open(output_image_path, "rb") as img_file:
            image_bytes = img_file.read()
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        
        # 임시 파일 정리
        try:
            os.remove(temp_file.name)
            os.remove(output_image_path)
        except Exception as e:
            print(f"⚠️ 파일 삭제 실패: {e}")
        
        return {
            "output_image": image_base64,
            "seed": generated_seed,
            "status": "success"
        }
        
    except Exception as e:
        return {
            "error": f"Inference failed: {str(e)}",
            "status": "error"
        }

# 서버 시작 시 모델 초기화
print("🔄 Initializing ACE Plus model...")
print(f"📁 Working directory: {os.getcwd()}")
print(f"📁 Files in current directory: {os.listdir('.')}")

try:
    initialize_model()
    print("🎉 Server is ready to handle requests!")
except Exception as e:
    import traceback
    print(f"❌ Model initialization failed: {e}")
    print(f"🔍 Full traceback: {traceback.format_exc()}")
    initialization_complete = False

# RunPod 서버리스 시작
print("🚀 Starting RunPod serverless handler...")
runpod.serverless.start({"handler": handler})
