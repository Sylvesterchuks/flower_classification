
import os
import argparse
import torch
import pandas as pd
import numpy as np
import json
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
# import albumentations as A
# from albumentations.pytorch import ToTensorV2


stats = (0.4862, 0.4561, 0.3941), (0.2202, 0.2142, 0.2160)

model_tsfm = T.Compose([
                T.Resize((224, 224)),
                # A.Normalize(*stats),
                T.ToTensor()             
             ])


with open('cat_to_name.json', 'r') as f:
    cat_dict = json.load(f)

cat_to_name = pd.Series(cat_dict)
cat_to_name.index = cat_to_name.index.astype(np.int32)
cat_to_name.sort_index(inplace=True)
classes = cat_to_name.values


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
   
    parser.add_argument('-i', '--Image',
        help="input image path", required=True)

    args = vars(parser.parse_args())  
    print(args)
    img_path = args['Image'] 
    #plt.imshow(get_image(img_path, model_tsfm).permute(1,2,0))
    #img_pred = eff_b2(get_image(img_path, model_tsfm).unsqueeze(0).to(device))
    #print(img_pred)
    #img_class = torch.argmax(img_pred)
    #print(img_class)
    #print(classes[img_class.item()])
