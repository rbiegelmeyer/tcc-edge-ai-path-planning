import os
model_folder_path = './results/result_W064xH064_D01_S000000_E005000'

# model_folder_path = os.path.abspath(model_folder_path)
ai_models = [f for f in os.listdir(model_folder_path) if f.endswith(('.h5', '.tflite', '.onnx'))]

# remove not file
ai_models = [f for f in ai_models if os.path.isfile(os.path.join(model_folder_path, f))]

print(ai_models)