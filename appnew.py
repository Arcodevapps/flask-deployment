import json
from io import BytesIO
from typing import List, Tuple

import requests
import torchvision
from flask import Flask, jsonify, request
from PIL import Image
from torch import argmax
from torch import device as tdevice
from torch import inference_mode, load, nn, softmax
from torchvision import models, transforms

# Set device
device = "cpu"

def pred_and_plot_image(
    model: nn.Module,
    class_names: List[str],
    image_path: str,
    image_size: Tuple[int, int] = (224, 224),
    transform: torchvision.transforms = None,
    device: tdevice = device,
):
   # Example of fixing the image_path variable
    
    response = requests.get(image_path)
    response.raise_for_status()  # Raise an exception for HTTP errors
    # Open image from content
    
    try:
        image = Image.open(BytesIO(response.content))
    except Exception as e:
        print(f"Error opening image: {e}")
        return None
    # Create transformation for image (if one doesn't exist)
    if transform is not None:
        image_transform = transform
    else:
        image_transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    ### Predict on image ###
    print(type(image))  # Verify the type of img
    print(image.size)   # Verify the size of img
    image.show()
    # Make sure the model is on the target device
    
    print("pass7")
    # Turn on model evaluation mode and inference mode
    model.eval()
    with inference_mode():
        # Transform and add an extra dimension to image (model requires samples in [batch_size, color_channels, height, width])
        transformed_image = image_transform(image)
        print(transformed_image.shape)
        transformed_image=transformed_image.unsqueeze(dim=0)
        # Make a prediction on image with an extra dimension and send it to the target device
        target_image_pred = model(transformed_image.to(device))

    # Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
    target_image_pred_probs = softmax(target_image_pred, dim=1)

    # Convert prediction probabilities -> prediction labels
    target_image_pred_label = argmax(target_image_pred_probs, dim=1)

  
    print(f"Pred: {class_names[target_image_pred_label]} | Prob: {target_image_pred_probs.max():.3f}")
    return (f"Pred: {class_names[target_image_pred_label]} | Prob: {target_image_pred_probs.max():.3f}")
#########################################################
# 1. Get pretrained weights for ViT-Base
pretrained_vit_weights = models.ViT_B_16_Weights.DEFAULT 

# 2. Setup a ViT model instance with pretrained weights
pretrained_vit = models.vit_b_16(weights=pretrained_vit_weights).to(device)

for parameter in pretrained_vit.parameters():
    parameter.requires_grad = False
    
# 4. Change the classifier head 
class_names = ['arecanut_budroot',
		  'healthy_arecnut',
		  'inflorecence',
		  'koleroga',
		  'leafspot',
		  'natural_aging',
		  'suity_mold',
		  'yellow_leaf']

##################################################################
pretrained_vit.heads = nn.Linear(in_features=768, out_features=len(class_names)).to(device)

pretrained_vit.load_state_dict(load("./vit_model_state_dict.pth",map_location=tdevice('cpu')))

#################################################################



##############################################################
app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    image_path = data["image_path"]
    
    output = pred_and_plot_image(model=pretrained_vit, image_path=image_path, class_names=class_names)
    
    return jsonify(output)

@app.route('/')
def hello():
    return "output<h1>Testing</h1>"
    