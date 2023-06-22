
### 1. Imports and class names setup ###
import gradio as gr
import os
import torchvision.transforms as T

from model import FlowerClassificationModel
from timeit import default_timer as timer
from typing import Tuple, Dict
from data_setup import classes, model_tsfm
from utils import *

# Setup class names
#class_names = ['pizza', 'steak', 'sushi']

### 2. Model and transforms preparation ###
#test_tsfm = T.Compose([T.Resize((224,224)),
#                        T.ToTensor(),
#                       T.Normalize(mean=[0.485, 0.456, 0.406], # 3. A mean of [0.485, 0.456, 0.406] (across each colour channel)
#                         std=[0.229, 0.224, 0.225]) # 4. A standard deviation of [0.229, 0.224, 0.225] (across each colour channel),
#                       ])

# Create ResNet50 Model
flower_model = FlowerClassificationModel(num_classes=len(classes), pretrained=True)

saved_path = 'flower_model_29.pth'

print('Loading Model State Dictionary')
# Load saved weights
flower_model.load_state_dict(
                torch.load(f=saved_path,
                           map_location=torch.device('cpu'), # load to CPU
                          )['model_state_dict']
                        )

print('Model Loaded ...')
### 3. Predict function ###

# Create predict function
from typing import Tuple, Dict

def predict(img) -> Tuple[Dict, float]:
    """Transforms and performs a prediction on img and returns prediction and time taken.
    """
    # Start the timer
    start_time = timer()
    
    # Transform the target image and add a batch dimension
    #img = get_image(img_path, model_tsfm).unsqueeze(0)
    img = model_tsfm(img)
    img = img.unsqueeze(0)

    # Put model into evaluation mode and turn on inference mode
    flower_model.eval()
    with torch.inference_mode():
        # Pass the transformed image through the model and turn the prediction logits into prediction probabilities
        pred_probs = torch.softmax(flower_model(img), dim=1)
    
    # Create a prediction label and prediction probability dictionary for each prediction class (this is the required format for Gradio's output parameter)
    pred_labels_and_probs = {classes[i]: float(pred_probs[0][i]) for i in range(len(classes))}

    # Calculate the prediction time
    pred_time = round(timer() - start_time, 5)
    
    # Return the prediction dictionary and prediction time 
    return pred_labels_and_probs, pred_time

### 4. Gradio App ###

# Create title, description and article strings
title= 'United Kingdom Flower Classification Mini 🌻🌼🌸❀💐🌷'
description = "An ResNet50 computer vision model to classify images of Flower Categories."
article = "<p>Flower Classification Created by Chukwuka </p><p style='text-align: center'><a href='https://github.com/Sylvesterchuks/dogbreed_app'>Github Repo</a></p>"


# Create examples list from "examples/" directory
example_list = [["examples/" + example] for example in os.listdir("examples")]

# Create the Gradio demo
demo = gr.Interface(fn=predict, # mapping function from input to output
                    inputs=gr.Image(type='pil'), # What are the inputs?
                    outputs=[gr.Label(num_top_classes=5, label="Predictions"), # what are the outputs?
                             gr.Number(label='Prediction time (s)')], # Our fn has two outputs, therefore we have two outputs
                    examples=example_list,
                    title=title,
                    description=description,
                    article=article
                   )
# Launch the demo
print('Gradio Demo Launched')
demo.launch()

