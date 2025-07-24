python infer_lora.py \
  --instruction "A smiling girl in a neat police uniform, wearing a badge." \
  --input_reference_image assets/samples/portrait/human_1.jpg \
  --task_type portrait \
  --task_model models/model_zoo.yaml \
  --cfg_folder config \
  --save_path examples/outputs/portrait_human_1.jpg \
  --infer_type diffusers
