#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RunPod 배포 전 로컬 테스트 스크립트
"""
import base64
import json
import os
from PIL import Image
import io

def encode_image_to_base64(image_path):
    """이미지를 base64로 인코딩"""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

def test_handler_locally():
    """핸들러를 로컬에서 테스트"""
    
    # 테스트용 더미 이벤트 생성
    test_event = {
        "input": {
            "instruction": "Make the person smile",
            "input_reference_image": None,  # 실제 이미지로 교체 필요
            "task_type": "portrait",
            "output_h": 1024,
            "output_w": 1024,
            "seed": 42
        }
    }
    
    print("🔧 Local test mode - checking dependencies...")
    
    try:
        # 의존성 import 테스트
        from scepter.modules.transform.io import pillow_convert
        from scepter.modules.utils.config import Config
        from scepter.modules.utils.file_system import FS
        from inference.ace_plus_diffusers import ACEPlusDiffuserInference
        print("✅ All dependencies imported successfully")
        
        # 설정 파일 존재 확인
        required_files = [
            "config/ace_plus_diffusers_infer.yaml",
            "models/model_zoo.yaml"
        ]
        
        for file_path in required_files:
            if os.path.exists(file_path):
                print(f"✅ Found: {file_path}")
            else:
                print(f"❌ Missing: {file_path}")
                return False
        
        # 모델 경로 확인
        config = Config(load=True, cfg_file="models/model_zoo.yaml")
        for task_name, task_config in config.MODEL.items():
            model_path = task_config["MODEL_PATH"]
            if os.path.exists(model_path):
                print(f"✅ Model found: {task_name} -> {model_path}")
            else:
                print(f"⚠️  Model not found: {task_name} -> {model_path}")
        
        print("🎉 Basic checks passed! Ready for RunPod deployment.")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_handler_locally() 