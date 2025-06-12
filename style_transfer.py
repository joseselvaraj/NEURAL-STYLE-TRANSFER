import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt

# Set device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image size (smaller = faster)
image_size = 256

# Load and preprocess image
def load_image(img_path, max_size=image_size):
    image = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((max_size, max_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)
    return image.to(device)

# Deprocess tensor to image
def im_convert(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.squeeze(0)
    image = image * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + \
            torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    image = image.clamp(0, 1)
    return transforms.ToPILImage()(image)

# Open file dialog to choose images
def choose_file(title):
    root = tk.Tk()
    root.withdraw()
    return filedialog.askopenfilename(title=title)

# Load VGG19 and freeze parameters
vgg = models.vgg19(pretrained=True).features.to(device).eval()
for param in vgg.parameters():
    param.requires_grad = False

# Define content and style layers
content_layers = ['conv_4']
style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

# Get features from layers
def get_features(image, model, layers=None):
    features = {}
    x = image
    i = 0
    for layer in model.children():
        x = layer(x)
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f'conv_{i}'
            if layers is None or name in layers:
                features[name] = x
    return features

# Gram matrix for style
def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram

# Main style transfer function
def run_style_transfer(content, style, num_steps=50, style_weight=1e6, content_weight=1):
    target = content.clone().requires_grad_(True).to(device)
    optimizer = optim.Adam([target], lr=0.003)

    content_features = get_features(content, vgg, content_layers)
    style_features = get_features(style, vgg, style_layers)
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

    for step in range(1, num_steps + 1):
        target_features = get_features(target, vgg)
        content_loss = torch.mean((target_features['conv_4'] - content_features['conv_4'])**2)

        style_loss = 0
        for layer in style_layers:
            target_feature = target_features[layer]
            target_gram = gram_matrix(target_feature)
            style_gram = style_grams[layer]
            layer_style_loss = torch.mean((target_gram - style_gram)**2)
            style_loss += layer_style_loss / (target_feature.shape[1] ** 2)

        total_loss = content_weight * content_loss + style_weight * style_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        print(f"Step {step}, Total loss: {total_loss.item():.2f}")

    return target

# === RUN ===
print("Choose content image:")
content_path = choose_file("Select Content Image")
print("Choose style image:")
style_path = choose_file("Select Style Image")

content = load_image(content_path)
style = load_image(style_path)

output = run_style_transfer(content, style, num_steps=50)

# Show the output
output_image = im_convert(output)
output_image.show()

# Save result
output_image.save("output_styled.png")
print("✅ Style transfer completed. Output saved as 'output_styled.png'.")
