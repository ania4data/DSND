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

arch='alexnet'   #ok  #'vgg19_bn' #run time    #densenet161' ok      #'resnet152' ok                #'inception_v3'  tensor open issue
node_hidden=512
num_epoch=1
epochs=num_epoch
learning_rate=learn_rate

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



if(arch=='resnet152'):
    model = models.resnet152(pretrained=True)
    input_size = 2048
    hidden_sizes = [node_hidden]
    output_size = 102

    for param in model.parameters():
        param.requires_grad = False


    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(input_size, hidden_sizes[0])),
                              ('relu1', nn.ReLU()),
                              ('fc2', nn.Linear(hidden_sizes[0], output_size)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

    model.fc = classifier   #last layer of resnet called fc not classifier so removing model.classifier with model.fc

if(arch=='alexnet'):
    model = models.alexnet(pretrained=True)
    input_size = 9216
    hidden_sizes = [node_hidden]
    output_size = 102

    for param in model.parameters():
        param.requires_grad = False


    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(input_size, hidden_sizes[0])),
                              ('relu1', nn.ReLU()),
                              ('fc2', nn.Linear(hidden_sizes[0], output_size)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

    model.classifier = classifier   #last layer of resnet called fc not classifier so removing model.classifier with model.fc
            
# if(arch=='inception_v3'):
#     model = models.inception_v3(pretrained=True)
#     input_size = 2048
#     hidden_sizes = [node_hidden]
#     output_size = 102

#     for param in model.parameters():
#         param.requires_grad = False


#     classifier = nn.Sequential(OrderedDict([
#                               ('fc1', nn.Linear(input_size, hidden_sizes[0])),
#                               ('relu1', nn.ReLU()),
#                               ('fc2', nn.Linear(hidden_sizes[0], output_size)),
#                               ('output', nn.LogSoftmax(dim=1))
#                               ]))

#     model.fc = classifier   #last layer of resnet called fc not classifier so removing model.classifier with model.fc

if(arch=='densenet161'):
    model = models.densenet161(pretrained=True)
    input_size = 2208
    hidden_sizes = [node_hidden]
    output_size = 102

    for param in model.parameters():
        param.requires_grad = False


    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(input_size, hidden_sizes[0])),
                              ('relu1', nn.ReLU()),
                              ('fc2', nn.Linear(hidden_sizes[0], output_size)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

    model.classifier = classifier   #last layer of resnet called fc not classifier so removing model.classifier with model.fc


print(classifier)
criterion = nn.NLLLoss()


# Only train the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(classifier.parameters(), lr=learning_rate) #, betas=(0.9, 0.999), eps=1e-8,
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




training(model, trainloader, evalloader, criterion, optimizer,device,epochs, 40)                    


test_loss, test_accuracy=validation(model, testloader, criterion,device)
print("test_loss: %f , test_accuracy : %f " % (test_loss, test_accuracy))


checkpoint = {'input_size': input_size,
              'output_size': 102,
              'hidden_layers': node_hidden,
              'model_name':arch,
              'optimizer_name':'Adam',
              'model_base':model,
              'model_classifier':classifier,  #, model.fc
              'optimizer':optimizer,
              'model_state_dict': model.state_dict(),'optimizer_state_dict':optimizer.state_dict,
              'optimizer_state_dict_':optimizer.state_dict(),'criterion_loss_function':criterion,
              'epochs':1,'printing_batch':40,'device':str(device),
              'class_to_label': trainset.class_to_idx, 'label_to_name':label_to_name}  


print('____________')         
#print(model.fc)
print('____________') 
print(node_hidden)
print('____________') 
print(arch)
print('____________') 
print(criterion)
print('____________') 
print(input_size)

torch.save(checkpoint, 'test6.pth')
