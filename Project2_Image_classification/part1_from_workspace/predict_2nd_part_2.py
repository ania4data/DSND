# Imports here

import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms,utils, models
from torch.utils.data import Dataset, DataLoader
import helper

import matplotlib.pyplot as plt
from collections import OrderedDict

import json
from PIL import Image

#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'

from workspace_utils import active_session


    
checkpoint_path='checkpoint_adam_resnet152_10epoch_opti_name.pth'
image_path='flowers/test/35/image_06984.jpg'
top_num=10
topk=top_num
#json file needed 
#cpu vs. gpu neede
device='cuda'


with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    


def label_to_name_function(label_,cat_to_name,label_to_class):
    
    class_=str(label_to_class[label_.item()])
    name_=cat_to_name[class_]
    return name_



def load_checkpoint(filepath):
    
    checkpoint = torch.load(filepath)
    model=checkpoint['model_base']
    #for param in model.parameters():
    #    param.requires_grad = False   #have to turn this of if I want to load optimizer as a whole in the next step

    model.fc = checkpoint['model_classifier']
    model.load_state_dict(checkpoint['model_state_dict'])             #keeping weight and layers
    model.class_to_idx=checkpoint['class_to_label']
    model=model.cuda()
    model=model.eval()
    return model,checkpoint
    

def process_image(filepath):
    im = Image.open(filepath)

    ratio=im.width/im.height

    if(ratio<1.0):
        new_height=int(256/ratio)
        im_resize=im.resize((256,new_height))
    else:
        new_width=int(256*ratio)
        im_resize=im.resize((new_width,256))

 

    center_y=int(im_resize.width/2.)
    center_x=int(im_resize.height/2.)
    #print(center_x,center_y)


    upper=center_y-112   # PIL  real x,is not actually along width, width->y   height->x

    left=center_x-112

    lower=center_y+112

    right=center_x+112



    im_resize_crop=im_resize.crop((upper,left,lower,right))

    np_image = np.array(im_resize_crop)

    np_image_norm=np_image/255.


    mean_=[0.485, 0.456, 0.406]
    std_=[0.229, 0.224, 0.225]

    np_image_norm_transform=np.zeros((224, 224,3))

    for i in range(3):

        np_image_norm_transform[:,:,i]=(np_image_norm[:,:,i]-mean_[i])/std_[i]



    np_image_norm_transform_=np_image_norm_transform.transpose((2, 0, 1))
    return np_image_norm_transform_



def predict(image_path, checkpoint_path, topk=top_num):
    
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    model_,checkpoint_ = load_checkpoint(checkpoint_path)
    device_=checkpoint_['device']
    
    np_image_norm_transform_=process_image(image_path)
    torch_img=torch.from_numpy(np_image_norm_transform_)
    title_=str(cat_to_name[image_path.split('/')[2]])    #for image title
    class__=int(image_path.split('/')[2])
    inputs=torch_img
    inputs=inputs.type_as(torch.FloatTensor())
    inputs=inputs.to(device_)



    model_.eval()
    with torch.no_grad():
        img=inputs
        img.unsqueeze_(0)  #   with single image batch size missing, so need to add another dimension  
                           #   https://discuss.pytorch.org/t/expected-stride-to-be-a-single-integer-value-or-a-list/17612/4
        output = model_.forward(img)
        out_exp = torch.exp(output)
        pred=out_exp.max(1)[1]
        pred_list=out_exp



    pred_numpy=pred_list.cpu().numpy()   #need to make to cpu 

    sort_list=np.argsort(pred_numpy[0])
    sort_list_rev=sort_list[::-1]

    top_k_list=sort_list[-topk:]
    top_k_list_rev=sort_list_rev[0:topk]
    
    top_k_list_rev=np.array(top_k_list_rev)

    class_to_label=checkpoint_['class_to_label']

    label_to_class = {v: k for k, v in class_to_label.items()}


    dict_= checkpoint_['label_to_name']
    dict__=label_to_class
    step=0

    flower_name_list_topk=[]
    flower_class_list_topk=[]
    probability_list_topk=[]

    for i in top_k_list_rev:
        step +=1

        flower_name_list_topk.append(dict_[i])
        flower_class_list_topk.append(dict__[i])
        probability_list_topk.append(pred_numpy[0][i])

    return title_,class__,flower_name_list_topk,flower_class_list_topk,probability_list_topk,topk  


def plot_flower_probability(image_path,checkpoint_path,topk):
    
    title_,class__,flower_name_list_topk,flower_class_list_topk,probability_list_topk,topk=predict(image_path, checkpoint_path, topk)

    flower_name_list_topk=flower_name_list_topk[::-1]
    probability_list_topk=probability_list_topk[::-1]
    flower_tuple_name=tuple(flower_name_list_topk)
    
    np_image_norm_transform_=process_image(image_path)
    torch_img=torch.from_numpy(np_image_norm_transform_)
    title_=str(cat_to_name[image_path.split('/')[2]])    #for image title

    ax=imshow(torch_img,title=title_)
    ax.set_title(title_)
    
    fig2, ax2 = plt.subplots()
    ind = np.arange(1, topk+1)
    plt.barh(ind, probability_list_topk)
    plt.yticks(ind,flower_tuple_name)
    plt.show()
    

    
title_,class__,flower_name_list_topk,flower_class_list_topk,probability_list_topk,topk_=predict(image_path, checkpoint_path, topk=top_num)
#print(title_,class__,flower_name_list_topk,flower_class_list_topk,probability_list_topk,topk_)

print('==========')
print('')
print('Flower actual name:',title_) 
print('Flower actual class:',class__) 
print('')
print('===== Top %d =====' %(topk_))
print('List of flower prediction: name: %s' %(flower_name_list_topk)) 
print('List of flower prediction: class: %s' %(flower_class_list_topk)) 
print('List of flower prediction: probability: %s' %(probability_list_topk)) 

#plot_flower_probability(image_path,checkpoint_path,5)    


    