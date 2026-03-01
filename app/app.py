import onnxruntime as ort
from torchvision import transforms
from PIL import Image
import numpy as np
import gradio as gr
import os

class_names = ['bicycle', 'car', 'motorcycle']
labels_ru = {
    'bicycle': 'Велосипед 🚲',
    'car': 'Машина 🚗', 
    'motorcycle': 'Мотоцикл 🏍️'
}

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

base_dir = os.path.dirname(os.path.abspath(__file__))
onnx_path = os.path.join(base_dir, '..', 'notebooks', 'models', 'resnet18_bs16_lr0.0001.onnx')

ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

def predict(image):
    if image is None:
        return {"Ошибка": "Загрузите изображение!"}
    
    img = Image.fromarray(image.astype('uint8'), 'RGB')
    img = test_transforms(img).unsqueeze(0).numpy()

    ort_inputs = {ort_session.get_inputs()[0].name: img}
    ort_outs = ort_session.run(None, ort_inputs)
    
    logits = ort_outs[0][0]
    exp_logits = np.exp(logits - np.max(logits))
    probabilities = exp_logits / np.sum(exp_logits)

    result_dict = {}
    for i, class_name in enumerate(class_names):
        result_dict[labels_ru[class_name]] = float(probabilities[i])
    return result_dict

# CSS для оформления
custom_css = """
body {
    background: #edecfd;
}
.gradio-container {
    background: #edecfd;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    margin-top: 20px;
    margin-bottom: 20px;
}
.title {
    color: #1a1a1a !important;
    font-size: 28px !important;
    text-align: center;
    font-weight: bold;
    border-bottom: 3px solid #3b82f6;
    padding-bottom: 15px;
}
.description {
    color: #6b7280 !important;
    font-size: 16px !important;
    text-align: center;
    margin-bottom: 20px;
}
button.primary {
    background: #3b82f6 !important;
    border-radius: 8px;
}
"""

iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy", label="📸 Загрузите фото"),
    outputs=gr.Label(num_top_classes=3, label="🔍 Результат"),
    title="Классификатор Транспорта: Велосипед, Машина или Мотоцикл",
    description="Загрузите изображение",
    css=custom_css
)

iface.launch()