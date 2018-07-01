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

# with active_session():
#     # do long-running work here

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# TODO: Define your transforms for the training, validation, and testing sets
data_transforms_train = transforms.Compose([transforms.RandomRotation([-30,30]),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

data_transforms_test_eval = transforms.Compose([transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])    
                                                
# TODO: Load the datasets with ImageFolder

trainset = datasets.ImageFolder(train_dir, transform=data_transforms_train)
testset = datasets.ImageFolder(test_dir, transform=data_transforms_test_eval)
evalset = datasets.ImageFolder(valid_dir, transform=data_transforms_test_eval)   
                                                
                                                
# TODO: Using the image datasets and the trainforms, define the dataloaders

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64,shuffle=True)                                                
evalloader = torch.utils.data.DataLoader(evalset, batch_size=64,shuffle=True)  


with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
class_to_label=trainset.class_to_idx   
label_to_class = {v: k for k, v in class_to_label.items()}

images, labels = next(iter(trainloader))

print(labels[0].item())
print(images[0].size())

label_to_name = {k: cat_to_name[v] for k, v in label_to_class.items()}     #translate label out from loader to name

def label_to_name_function(label_,cat_to_name,label_to_class):
    
    class_=str(label_to_class[label_.item()])
    name_=cat_to_name[class_]
    return name_


#print(label_to_name)
#print(label_to_name_function(labels[0],cat_to_name,label_to_class))

# model = models.resnet152(pretrained=True)


# for param in model.parameters():
#     param.requires_grad = False
# print(model)

# TODO: Build and train your network

model = models.resnet152(pretrained=True)
input_size = 2048
hidden_sizes = [1024,512,256]
output_size = 102

for param in model.parameters():
    param.requires_grad = False


classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input_size, hidden_sizes[0])),
                          ('relu1', nn.ReLU()),
                          ('dropout1',nn.Dropout(p=0.2)),
                          ('fc2', nn.Linear(hidden_sizes[0], hidden_sizes[1])),
                          ('relu2', nn.ReLU()),
                          ('dropout2',nn.Dropout(p=0.2)),
                          ('fc3', nn.Linear(hidden_sizes[1], hidden_sizes[2])),
                          ('relu3', nn.ReLU()),
                          ('dropout3',nn.Dropout(p=0.1)),
                          ('fc4', nn.Linear(hidden_sizes[2], output_size)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
        
model.fc = classifier   #last layer of resnet called fc not classifier so removing model.classifier with model.fc

criterion = nn.NLLLoss()

# Only train the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(model.fc.parameters(), lr=1e-3) #, betas=(0.9, 0.999), eps=1e-8,
# print(model.fc)
# print(optimizer)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def training(model, trainloader, evalloader, criterion, optimizer,device,epochs, print_batch):
    
    model=model
    trainloader=trainloader
    evalloader=evalloader
    criterion=criterion
    optimizer=optimizer
    device=device
    
    model.to(device)

    
    
    for e in range(epochs):
        
        training_loss_running = 0.0
        accuracy_running=0.0
        count__=0
        
        model.train()   #with dropouts
        for step, (inputs, labels) in enumerate(trainloader):
                        
            inputs=inputs.to(device)
            labels=labels.to(device)
                        
            optimizer.zero_grad()
            
            output = model.forward(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            out_exp = torch.exp(output)
            
            count_vec = (labels.data == out_exp.max(1)[1])  #true count vector
            count__ +=labels.size(0)
            
            accuracy_running += count_vec.type_as(torch.FloatTensor()).sum().item()
            
            training_loss_running += loss.item()

            if step % print_batch == 0:
                
                              
                #gradient turned off and model_eval inside of validation function
                test_loss, test_accuracy = validation(model, evalloader, criterion,device)  
                
#                 print("epoch %d / %d, accuracy_run_train: %f, loss_run_train: %f, eval_accuracy: %f, eval_loss: %f"
#                       % (e+1,epochs,accuracy_running/print_batch,training_loss_running/print_batch,
#                         test_accuracy,test_loss))

                print("epoch %d / %d, accuracy_run_train: %f, loss_run_train: %f, eval_accuracy: %f, eval_loss: %f"
                      % (e+1,epochs,accuracy_running*100/count__,training_loss_running/count__,
                        test_accuracy,test_loss))                
                
                accuracy_running=0.0
                training_loss_running=0.0
                count__=0.0
                # turn on drop out grad for training
                model.train()

def validation(model, loader_, criterion,device):
    
    model=model
    loader_=loader_
    criterion=criterion
    device=device
    count_=0
    model.eval()   #evaluation mode to only do forward without dropout
    
    with torch.no_grad():
        
        accuracy = 0
        test_loss = 0
        for step, (inputs, labels) in enumerate(loader_):
            
            inputs=inputs.to(device)
            labels=labels.to(device)
            
            output = model.forward(inputs)
            test_loss += criterion(output, labels).item()

            out_exp = torch.exp(output)    #for nn.NLLLoss()
            
            count_vec = (labels.data == out_exp.max(1)[1])
            count_ +=labels.size(0)
            # Accuracy is number of correct predictions divided by all predictions, just take the mean (no is not, you need to divide by sample size here to be called accuracy)
            accuracy += count_vec.type_as(torch.FloatTensor()).sum().item()  #count of true
            
        test_loss=test_loss/(count_)
        accuracy=accuracy*100.0/(count_)
        
    return test_loss, accuracy


#training(model, trainloader, evalloader, criterion, optimizer,device,1, 40)                    


#test_loss, test_accuracy=validation(model, testloader, criterion,device)
#print("test_loss: %f , test_accuracy : %f " % (test_loss, test_accuracy))


checkpoint = {'input_size': 2048,
              'output_size': 102,
              'hidden_layers': [1024,512,256],
              'model_name':'resnet152',
              'optimizer_name':'Adam',
              'model_base':model,
              'model_classifier':model.fc,
              'optimizer':optimizer,
              'model_state_dict': model.state_dict(),'optimizer_state_dict':optimizer.state_dict,
              'optimizer_state_dict_':optimizer.state_dict(),'criterion_loss_function':criterion,
              'epochs':1,'printing_batch':40,'device':'cuda',
              'class_to_label': trainset.class_to_idx, 'label_to_name':label_to_name}   

# "state_dic" keep the weights of the model only for a specific structure
# 'class_to_label' keep conversion to 
# 'label_to_name' use json file to get the name of label (by using class_to_idx)
# 'optimizer_state_dict' general shape of optimizer
# 'optimizer_state_dict()' content of optimizer
# 'epochs' number of epochs run
# 'printing_batch' how often print
torch.save(checkpoint, 'test.pth')

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
    #im.rotate(45).show()
    #im.show()
    #print(im.width)
    ratio=im.width/im.height
    #print(ratio)
    if(ratio<1.0):
        new_height=int(256/ratio)
        im_resize=im.resize((256,new_height))
    else:
        new_width=int(256*ratio)
        im_resize=im.resize((new_width,256))

    #im_resize.show()    

    center_y=int(im_resize.width/2.)
    center_x=int(im_resize.height/2.)
    #print(center_x,center_y)


    upper=center_y-112   # PIL  real x,is not actually along width, width->y   height->x

    left=center_x-112

    lower=center_y+112

    right=center_x+112



    im_resize_crop=im_resize.crop((upper,left,lower,right))
    #img.resize(size
    #im_resize_crop.show()

    np_image = np.array(im_resize_crop)
    #print(np_image)
    np_image_norm=np_image/255.
    #print(np_image_norm)
    #np.shape(np_image_norm)

    mean_=[0.485, 0.456, 0.406]
    std_=[0.229, 0.224, 0.225]

    np_image_norm_transform=np.zeros((224, 224,3))

    for i in range(3):

        np_image_norm_transform[:,:,i]=(np_image_norm[:,:,i]-mean_[i])/std_[i]

    #plt.hist(np.ravel(np_image_norm_transform[:,:,0]))
    #plt.hist(np.ravel(np_image_norm_transform[:,:,1]))
    #plt.hist(np.ravel(np_image_norm_transform[:,:,2]))

    np_image_norm_transform_=np_image_norm_transform.transpose((2, 0, 1))
    return np_image_norm_transform_

#Dir_ = 'flowers/test/20/image_04910.jpg'

#np_image_norm_transform_=process_image(Dir_)
#torch_img=torch.from_numpy(np_image_norm_transform_)
#print(np.shape(np_image_norm_transform_))



def predict(image_path, checkpoint_path, topk=5):
    
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    model_,checkpoint_ = load_checkpoint(checkpoint_path)
    device_=checkpoint_['device']
    
    np_image_norm_transform_=process_image(image_path)
    torch_img=torch.from_numpy(np_image_norm_transform_)
    title_=str(cat_to_name[image_path.split('/')[2]])    #for image title
    class__=int(image_path.split('/')[2])
    #ax=imshow(torch_img,title=title_)
    #ax.set_title(title_)
    inputs=torch_img
    inputs=inputs.type_as(torch.FloatTensor())
    inputs=inputs.to(device_)
    #print(inputs.size())


    model_.eval()
    with torch.no_grad():
        img=inputs
        img.unsqueeze_(0)  #   with single image batch size missing, so need to add another dimension  
                           #   https://discuss.pytorch.org/t/expected-stride-to-be-a-single-integer-value-or-a-list/17612/4
        output = model_.forward(img)
        out_exp = torch.exp(output)
        pred=out_exp.max(1)[1]
        pred_list=out_exp

    #print('pred',pred)
    #print(out_exp.max(1))
    #print(pred_list)

    pred_numpy=pred_list.cpu().numpy()   #need to make to cpu 
    #print(pred_numpy)
    sort_list=np.argsort(pred_numpy[0])
    sort_list_rev=sort_list[::-1]
    #print('sum of probability',pred_numpy[0].sum(),pred_numpy[0].max(),pred_numpy[0].min())
    top_k_list=sort_list[-topk:]
    top_k_list_rev=sort_list_rev[0:topk]
    
    top_k_list_rev=np.array(top_k_list_rev)

    class_to_label=checkpoint_['class_to_label']

    label_to_class = {v: k for k, v in class_to_label.items()}

    #print(top_k_list)
    #print(top_k_list_rev)
    #print(checkpoint_['label_to_name'])
    dict_= checkpoint_['label_to_name']
    dict__=label_to_class
    step=0
    #print(' ')
    #print('REAL-----',dict_[label[0].item()]) 
    flower_name_list_topk=[]
    flower_class_list_topk=[]
    probability_list_topk=[]

    for i in top_k_list_rev:
        step +=1
        #print('Rank',step,dict_[i],i,pred_numpy[0][i])
        flower_name_list_topk.append(dict_[i])
        flower_class_list_topk.append(dict__[i])
        probability_list_topk.append(pred_numpy[0][i])

    #fig, ax = plt.subplots()
    #ind = np.arange(1, 6)

    #print(flower_name_list_topk)
    #print(probability_list_topk)
    #plt.bar(flower_name_list_topk, probability_list_topk)
    #plt.show()
    ##fig2, ax2 = plt.subplots()
    #flower_name_list_topk=flower_name_list_topk[::-1]
    #probability_list_topk=probability_list_topk[::-1]
    #flower_tuple_name=tuple(flower_name_list_topk)
    ##ind = np.arange(1, topk+1)
    ##plt.barh(ind, probability_list_topk)
    #ax.set_xticks(ind)
    #ax.set_xticklabels(flower_name_list_topk)
    ##plt.yticks(ind,flower_tuple_name)
    ##plt.show()
    return title_,class__,flower_name_list_topk,flower_class_list_topk,probability_list_topk,topk  #this is added based on rubic, I have matplot lib inside though


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
    
    
checkpoint_path='checkpoint_adam_resnet152_10epoch_opti_name.pth'
image_path='flowers/test/35/image_06984.jpg'
#predict(image_path, checkpoint_path, topk=5)  
    
print(predict(image_path, checkpoint_path, topk=5))    
#plot_flower_probability(image_path,checkpoint_path,5)    

    