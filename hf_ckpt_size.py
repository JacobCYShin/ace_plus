from huggingface_hub import HfApi

def get_model_total_size(repo_id):
    api = HfApi()
    try:
        model_info = api.model_info(repo_id=repo_id, files_metadata=True)
        total_size_bytes = 0
        print(f"모델 '{repo_id}'의 파일 목록 및 크기:")
        for sibling in model_info.siblings:
            if sibling.size is not None:
                total_size_bytes += sibling.size
                print(f"  - {sibling.rfilename}: {sibling.size / (1024**2):.2f} MB")
            else:
                print(f"  - {sibling.rfilename}: 크기 정보 없음")

        total_size_gb = total_size_bytes / (1024**3)
        print(f"\n총 모델 크기: {total_size_gb:.2f} GB")
        return total_size_gb

    except Exception as e:
        print(f"모델 정보를 가져오는 중 오류 발생: {e}")
        return None

# 예시 사용
model_name = "black-forest-labs/FLUX.1-Fill-dev" # 확인하고 싶은 모델 ID
get_model_total_size(model_name)

# SDXL Base (예시, 실제 모델 ID는 다를 수 있음)
# model_name_sdxl = "stabilityai/stable-diffusion-xl-base-1.0"
# get_model_total_size(model_name_sdxl)