from matplotlib import cm
import numpy as np
from pathlib import Path
import PIL.Image
import torch

from src.models.dino import DINOSegmentationModel
from src.models.vit import ViTSegmentation
from src.models.unet import UNet
from src.utils import get_transform


color_map = cm.get_cmap("tab20", 9)
fixed_colors = np.array([color_map(i)[:3] for i in range(9)]) * 255


def mask_to_color(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_idx in range(9):
        color_mask[mask == class_idx] = fixed_colors[class_idx]
    return color_mask


def add_column_labels(image: PIL.Image, labels: list[str], width: int, label_height: int) -> PIL.Image:
    new_height = image.height + label_height
    new_image = PIL.Image.new("RGB", (image.width, new_height), color=(0, 0, 0))
    new_image.paste(image, (0, label_height))
    draw = PIL.ImageDraw.Draw(new_image)
    font_size = 16
    try:
        font = PIL.ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = PIL.ImageFont.load_default()
    for i, label in enumerate(labels):
        text_position = ((i + 1) * width - width // 2, label_height // 2)
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        centered_position = (text_position[0] - text_width // 2, text_position[1] - text_height // 2)
        draw.text(centered_position, label, fill="white", font=font)
    return new_image


def main() -> None:
    images_path = "assets/images_examples"
    weights_path = "weights"
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    height, width = 224, 224
    label_height = 40

    model_weight1 = "weights/dino.pth"
    model_weight2 = "weights/vit.pth"
    models_weights = ["weights/dino.pth", "weights/vit.pth", "weights/unet.pth"]
    model_weight3 = "weights/unet.pth"
    images = sorted(list(Path(images_path).glob("**/*.jpg")))
    num_images, num_models = len(images), len(models_weights)

    combined_height = height * num_images
    combined_width = width * (num_models + 1)
    combined_image = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
    device = torch.device("cpu")

    model1 = DINOSegmentationModel()
    model1.segmentation_head.load_state_dict(torch.load(model_weight1, map_location=device))
    model1.eval()
    model2 = ViTSegmentation()
    model2.segmentation_head.load_state_dict(torch.load(model_weight2, map_location=device))
    model2.eval()
    model3 = UNet()
    model3.load_state_dict(torch.load(model_weight3, map_location=device))
    model3.eval()
    for i, model_weight in enumerate(models_weights):
        if model_weight == model_weight1:
            model = model1
        elif model_weight == model_weight2:
            model = model2
        else:
            model = model3
        for m, img in enumerate(images):
            image = PIL.Image.open(img).convert("RGB")
            transform = get_transform(model.mean, model.std)
            image = transform(image)
            output = model(image.unsqueeze(0))
            _, mask = output.max(dim=1)
            mask = mask[0].cpu().numpy()
            mask = mask_to_color(mask)
            image = image.permute(1, 2, 0).numpy()
            image = (image * model.std + model.mean).clip(0, 1)
            image = (image * 255).astype(np.uint8)
            combined_image[m * height:(m+1) * height, :width, :] = image
            combined_image[m * height:(m+1) * height, (i + 1) * width:(i+2) * width, :] = mask

    image = PIL.Image.fromarray(combined_image)
    column_labels = ["Original"] + [w.split("/")[1].split(".")[0] for w in models_weights]
    image = add_column_labels(image, column_labels, width, label_height)
    image.save("assets/combine_image11.png")


if __name__ == "__main__":
    main()

