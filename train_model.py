# Import your dependencies.
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import os
import sys
import logging
import argparse

import smdebug.pytorch as smd
from smdebug import modes
from smdebug.profiler.utils import str2bool
from smdebug.pytorch import get_hook

from torchvision import datasets, transforms, models
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def test(model, test_loader, criterion, device, hook):
    model.eval()                  # for testing using evalualion function
    test_loss = 0                 # set testing loss
    correct = 0                   # set testing correct predictions count   

    hook.set_mode(smd.modes.EVAL) # set the debugger hook

    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        loss = criterion(output, target)
        _, preds = torch.max(output, 1)
        test_loss += loss.item() * data.size(0)
        correct += torch.sum(preds == target.data)

    total_loss = test_loss // len(test_loader)
    total_acc = correct.double() // len(test_loader)
    
    logger.info(
        "Test set: Average loss: {:.4f}, Accuracy: {}\n".format(
            total_loss, total_acc)
    )


def train(model, train_loader, validation_loader, criterion, optimizer, epoch, device, hook):
    best_loss = 1e6
    loss_counter = 0
    image_dataset={'train':train_loader, 'valid':validation_loader}
    
    for e in range(epoch):
        logger.info(f"Epoch:{e}")
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
                hook.set_mode(smd.modes.TRAIN) # set debugging hook
            else:
                model.eval()
                hook.set_mode(smd.modes.EVAL)

            current_loss = 0.0
            current_corrects = 0
            
            for data, target in image_dataset[phase]:
                data = data.to(device)
                target = target.to(device)
                outputs = model(data)
                loss = criterion(outputs, target)
                
                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                _, preds = torch.max(outputs, 1)
                current_loss += loss.item() * data.size(0)
                current_corrects += torch.sum(preds == target.data)
                
            epoch_loss = current_loss // len(image_dataset[phase])
            epoch_acc = current_corrects // len(image_dataset[phase])
            
            if phase == 'valid':
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                else:
                    loss_counter += 1
            
            logger.info(
                "{} loss: {:.4f}, acc: {:.4f}, best loss: {:.4f}\n".format(
                    phase, epoch_loss, epoch_acc, best_loss)
            )

        if loss_counter==1:
            break

    return model


def net():
    # ResNet50 
    resnet50 = models.resnet50(pretrained=True)

    # Since the model is pre-trained, there is no need to train it once again
    for param in resnet50.parameters():
        param.requires_grad = False # freeze the model
        convolutional_base_features = resnet50.fc.in_features

    # output = 133 classifications
    resnet50.fc = nn.Sequential(nn.Linear(convolutional_base_features, 256),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(256, 133))
     
    return resnet50


def create_data_loaders(data, batch_size):  
    test_data_path = os.path.join(data, 'test')
    valid_data_path=os.path.join(data, 'valid')
    train_data_path = os.path.join(data, 'train')
    
    test_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),])
    valid_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),])
    train_transform = transforms.Compose([transforms.RandomResizedCrop((224, 224)), transforms.RandomHorizontalFlip(), transforms.ToTensor(),])
                                                            
    # Load the train, test and validation data from S3 location
    test_data = datasets.ImageFolder(root=test_data_path, transform=test_transform)
    test_data_loader  = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    valid_data = datasets.ImageFolder(root=valid_data_path, transform=valid_transform)
    valid_data_loader  = torch.utils.data.DataLoader(valid_data, batch_size=batch_size) 
    
    train_data = datasets.ImageFolder(root=train_data_path, transform=train_transform)
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    return train_data_loader, test_data_loader, valid_data_loader


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on Device {device}")
    logger.info(f'Hyperparameters are LR: {args.lr}, Batch Size: {args.batch_size}, epoch: {args.epochs}')
    logger.info(f'Data Paths: {args.data_dir}')
    train_loader, test_loader, validation_loader=create_data_loaders(args.data_dir, args.batch_size)

    # Initialize our model by calling the net() function
    model=net()
    model=model.to(device)
    
    # Create the debugging hook
    hook = smd.Hook.create_from_json_file()
    hook.register_module(model)
    
    # Create your loss and optimizer
    loss_criterion = nn.CrossEntropyLoss(ignore_index=133)
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)
    hook.register_loss(loss_criterion)
    
    # Call the train function to start training our model
    epoch = args.epochs
    logger.info("Starting model training")
    model=train(model, train_loader, validation_loader, loss_criterion, optimizer, epoch, device, hook)
    
    # Test the model to see its accuracy
    logger.info("Starting model testing")
    test(model, test_loader, loss_criterion, device, hook)
    
    # Save the trained model to S3
    logger.info("Saving our model")
    torch.save(model.state_dict(), os.path.join(args.model_dir, "resnet50.pt"))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1.0,
        metavar="LR",
        help="learning rate (default: 1.0)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        metavar="N",
        help="number of epochs to train (default: 5)",
    )
    
    # Container environment
    parser.add_argument("--model-dir", type=str, default=os.environ['SM_MODEL_DIR'])    
    parser.add_argument("--data-dir", type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    
    args=parser.parse_args()
    
    main(args)
    



