import gradio as gr
import PIL.Image, PIL.ImageOps
import torch
import torchvision.transforms.functional as F
from matplotlib import cm
from matplotlib.colors import to_hex
import numpy as np

from src.models.dino import DINOSegmentationModel
from src.models.vit import ViTSegmentation
from src.models.unet import UNet
from src.utils import get_transform


device = torch.device("cpu")
model_weight1 = "weights/dino.pth"
model_weight2 = "weights/vit.pth"
model_weight3 = "weights/unet.pth"

model1 = DINOSegmentationModel()
model1.segmentation_head.load_state_dict(torch.load(model_weight1, map_location=device))
model1.eval()
model2 = ViTSegmentation()
model2.segmentation_head.load_state_dict(torch.load(model_weight2, map_location=device))
model2.eval()
model3 = UNet()
model3.load_state_dict(torch.load(model_weight3, map_location=device))
model3.eval()

mask_labels = {
    "0": "Background", "1": "Person", "2": "Skin", "3": "Left-brow", "4": "Right-brow",
    "5": "Left-eye", "6": "Right-eye", "7": "Lips", "8": "Teeth"
}

color_map = cm.get_cmap("tab20", 9)
label_colors = {label: to_hex(color_map(idx / len(mask_labels))[:3]) for idx, label in enumerate(mask_labels)}
fixed_colors = np.array([color_map(i)[:3] for i in range(9)]) * 255


def mask_to_color(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_idx in range(9):
        color_mask[mask == class_idx] = fixed_colors[class_idx]
    return color_mask


def segment_image(image, model_name: str) -> PIL.Image:
    if model_name == "DINO":
        model = model1
    elif model_name == "ViT":
        model = model2
    else:
        model = model3

    original_width, original_height = image.size
    transform = get_transform(model.mean, model.std)
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        mask = model(input_tensor)
    mask = torch.argmax(mask.squeeze(), dim=0).cpu().numpy()

    mask_image = mask_to_color(mask)

    mask_image = PIL.Image.fromarray(mask_image)
    mask_aspect_ratio = mask_image.width / mask_image.height

    new_height = original_height
    new_width = int(new_height * mask_aspect_ratio)
    mask_image = mask_image.resize((new_width, new_height), PIL.Image.Resampling.NEAREST)

    final_mask = PIL.Image.new("RGB", (original_width, original_height))
    offset = ((original_width - new_width) // 2, 0)
    final_mask.paste(mask_image, offset)

    return final_mask

def generate_legend_html_compact() -> str:
    legend_html = """
    <div style="display: flex; flex-wrap: wrap; gap: 10px; justify-content: center;">
    """
    for idx, (label, color) in enumerate(label_colors.items()):
        legend_html += f"""
        <div style="display: flex; align-items: center; justify-content: center; 
                     padding: 5px 10px; border: 1px solid {color}; 
                     background-color: {color}; border-radius: 5px; 
                     color: white; font-size: 12px; text-align: center;">
            {mask_labels[label]}
        </div>
        """
    legend_html += "</div>"
    return legend_html

examples = [
    ["assets/images_examples/image1.jpg"],
    ["assets/images_examples/image2.jpg"],
    ["assets/images_examples/image3.jpg"]
]

with gr.Blocks() as demo:
    gr.Markdown("## Face Segmentation")
    with gr.Row():
        with gr.Column():
            pic = gr.Image(label="Upload Human Image", type="pil", height=400, width=400)
            model_choice = gr.Dropdown(choices=["DINO", "ViT", "UNet"], label="Select Model", value="DINO")
            with gr.Row():
                with gr.Column(scale=1):
                    predict_btn = gr.Button("Predict")
                with gr.Column(scale=1):
                    clear_btn = gr.Button("Clear")

        with gr.Column():
            output = gr.Image(label="Mask", type="pil", height=400, width=400)
            legend = gr.HTML(label="Legend", value=generate_legend_html_compact())

    predict_btn.click(fn=segment_image, inputs=[pic, model_choice], outputs=output, api_name="predict")
    clear_btn.click(lambda: (None, None), outputs=[pic, output])
    gr.Examples(examples=examples, inputs=[pic])

demo.launch()
