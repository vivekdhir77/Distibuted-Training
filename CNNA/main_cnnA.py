import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import logging
import time
from datetime import datetime
import os
import glob


## for distributed training
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 
num_epochs = 10
batch_size = 16
learning_rate = 0.001
start = 0

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class ConvNet(nn.Module):
    '''Smaller CNN architecture'''
    def __init__(self):
        super(ConvNet, self).__init__()
        # Define layers
        self.conv1 = nn.Conv2d(3, 4, 3)  # Reduced number of filters and kernel size
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(4, 8, 3)  # Reduced number of filters

        # FC layers
        self.fc1 = nn.Linear(8 * 6 * 6, 32)  # Reduced number of neurons
        self.fc2 = nn.Linear(32, 10)  # Directly map to 10 classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # First conv+pool
        x = self.pool(F.relu(self.conv2(x)))  # Second conv+pool
        x = x.view(-1, 8 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def load_data(batch_size):
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                              shuffle=False, sampler=DistributedSampler(train_dataset, shuffle=True))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True) 
    return train_loader, test_loader

def get_latest_checkpoint(checkpoint_dir):
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'latest.pth'))
    if not checkpoint_files:
        return None
    return max(checkpoint_files, key=os.path.getctime)

def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, n_total_steps, initial_train_time=0):
    # For tracking metrics
    train_accuracies = []
    test_accuracies = []
    best_accuracy = 0.0
    total_train_time = initial_train_time  # Start with previously accumulated time
    
    for epoch in range(start, num_epochs):
        model.train()
        epoch_start_time = time.time()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # fwd pass
            # Check if model is wrapped in DistributedDataParallel
            if isinstance(model, DistributedDataParallel):
                outputs = model.module(images)
            else:
                outputs = model(images)
                
            loss = criterion(outputs, labels)

            # bwd pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate training accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()

            if (i+1) % 512 == 0:    # print every 512 mini-batches
                current_loss = loss.item()
                logging.info(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {current_loss:.4f}')
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {current_loss:.4f}')
        
        # Calculate epoch train accuracy
        train_accuracy = 100 * correct / total
        train_accuracies.append(train_accuracy)
        
        # Evaluate on test set
        test_accuracy = evaluate_accuracy(model, test_loader)
        test_accuracies.append(test_accuracy)
        
        # Update total training time
        epoch_time = time.time() - epoch_start_time
        total_train_time += epoch_time
        
        logging.info(f'Epoch {epoch+1} completed in {epoch_time:.2f} seconds')
        logging.info(f'Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%')
        logging.info(f'Total training time so far: {total_train_time:.2f} seconds')
        print(f'Epoch {epoch+1} completed in {epoch_time:.2f} seconds')
        print(f'Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%')
        print(f'Total training time so far: {total_train_time:.2f} seconds')
        
        # Update best accuracy
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            # Save best model
            if not distributed_training or (distributed_training and global_rank == 0):
                print(f"New best accuracy: {best_accuracy:.2f}%")
                PATH = f'./weights/best_model.pth'
                # Save the correct state dict
                if isinstance(model, DistributedDataParallel):
                    model_state_dict = model.module.state_dict()
                else:
                    model_state_dict = model.state_dict()
                    
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_state_dict,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_accuracy': train_accuracy,
                    'test_accuracy': test_accuracy,
                    'best_accuracy': best_accuracy,
                    'total_train_time': total_train_time
                }, PATH)
        
        # Save checkpoint after each epoch
        PATH = f'./weights/latest.pth'
        # Save the correct state dict
        if isinstance(model, DistributedDataParallel):
            model_state_dict = model.module.state_dict()
        else:
            model_state_dict = model.state_dict()
            
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'train_accuracies': train_accuracies,
            'test_accuracies': test_accuracies,
            'train_accuracy': train_accuracies[-1] if train_accuracies else 0,
            'test_accuracy': test_accuracies[-1] if test_accuracies else 0,
            'best_accuracy': best_accuracy,
            'total_train_time': total_train_time
        }, PATH)
        
    return train_accuracies, test_accuracies, best_accuracy, total_train_time

def evaluate_accuracy(model, test_loader):
    """Calculate accuracy on a dataset"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            if isinstance(model, DistributedDataParallel):
                outputs = model.module(images)
            else:
                outputs = model(images)
                
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

def evaluate_model(model, test_loader):
    confusion_matrix = torch.zeros(10, 10, dtype=torch.int64)
    
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        
        if isinstance(model, DistributedDataParallel):
            outputs = model.module(images)
        else:
            outputs = model(images)
            
        _, predicted = torch.max(outputs, 1)
        
        for t, p in zip(labels.view(-1), predicted.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1

    return confusion_matrix

def log_metrics(confusion_matrix):
    print('\nMetrics for each class:')
    print('------------------------')
    
    for i in range(10):
        TP = confusion_matrix[i, i].item()
        FP = (confusion_matrix[:, i].sum() - confusion_matrix[i, i]).item()
        FN = (confusion_matrix[i, :].sum() - confusion_matrix[i, i]).item()
        TN = (confusion_matrix.sum() - confusion_matrix[i, :].sum() - confusion_matrix[:, i].sum() + confusion_matrix[i, i]).item()

        accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) != 0 else 0
        precision = TP / (TP + FP) if (TP + FP) != 0 else 0
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) != 0 else 0

        print(f'\nClass: {classes[i]}')
        print(f'Accuracy: {accuracy:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1-score: {f1:.4f}')
        print(f'Specificity: {specificity:.4f}')

        log_message = f"""
Class: {classes[i]}
Accuracy: {accuracy:.4f}
Precision: {precision:.4f}
Recall: {recall:.4f}
F1-score: {f1:.4f}
Specificity: {specificity:.4f}"""
        logging.info(log_message)

if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("This code requires GPU support. No GPU found!")

    
    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    global_rank = int(os.environ.get('RANK', -1))
    
    # to determine if we're using distributed training
    distributed_training = local_rank != -1 and global_rank != -1
    
    if distributed_training:
        init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cuda:0')  # Use the first GPU if available

    os.makedirs('./weights', exist_ok=True)

    # data loading
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=transform)

    if distributed_training:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size,
            shuffle=False, sampler=train_sampler)
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size,
            shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True)

    model = ConvNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Load checkpoint if exists
    checkpoint_dir = './weights'
    latest_checkpoint = get_latest_checkpoint(checkpoint_dir)
    
    # Initialize tracking variables
    train_accuracies = []
    test_accuracies = []
    best_accuracy = 0.0
    total_train_time = 0.0
    
    if latest_checkpoint:
        try:
            if distributed_training:
                state = torch.load(latest_checkpoint, map_location=f'cuda:{local_rank}')
            else:
                state = torch.load(latest_checkpoint, map_location='cuda:0')
            
            model.load_state_dict(state['model_state_dict'])
            optimizer.load_state_dict(state['optimizer_state_dict'])
            start = state['epoch'] + 1  # Add 1 because we want to start from the next epoch
            
            # Load accuracy histories if available
            if 'train_accuracies' in state:
                train_accuracies = state['train_accuracies']
                print(f"Loaded {len(train_accuracies)} previous train accuracy records")
            
            if 'test_accuracies' in state:
                test_accuracies = state['test_accuracies']
                print(f"Loaded {len(test_accuracies)} previous test accuracy records")
                
            if 'best_accuracy' in state:
                best_accuracy = state['best_accuracy']
                print(f"Previous best accuracy: {best_accuracy:.4f}%")
                
            if 'total_train_time' in state:
                total_train_time = state['total_train_time']
                print(f"Previous total training time: {total_train_time:.2f} seconds")
                
            print(f"Loaded model from {latest_checkpoint}")
            print(f"Continuing training from epoch {start}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting training from scratch")
            start = 0
    else:
        print("No checkpoint found. Starting training from scratch")
        start = 0

    # Wrap model in DistributedDataParallel only if using distributed training
    if distributed_training:
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # Set up logging
    logging.basicConfig(
        filename=f'training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
        level=logging.INFO,
        format='%(asctime)s - %(message)s'
    )

    n_total_steps = len(train_loader)  # Define n_total_steps here

    # Before training loop
    start_time = time.time()
    train_accuracies, test_accuracies, best_accuracy, total_train_time = train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, n_total_steps, total_train_time)

    total_time = time.time() - start_time

    # Clean up distributed training resources if needed
    if distributed_training:
        destroy_process_group()
    logging.info(f'Training completed in {total_time:.2f} seconds')
    print('Finished Training')
    PATH = f'./weights/cnn_{batch_size}_{num_epochs}.pth'
    # Save the correct state dict depending on model type
    if isinstance(model, DistributedDataParallel):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
        
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'train_accuracies': train_accuracies,
        'test_accuracies': test_accuracies,
        'train_accuracy': train_accuracies[-1] if train_accuracies else 0,
        'test_accuracy': test_accuracies[-1] if test_accuracies else 0,
        'best_accuracy': best_accuracy,
        'total_train_time': total_train_time
    }, PATH)

    # Evaluate model
    confusion_matrix = evaluate_model(model, test_loader)
    
    # Only print metrics on the main process
    if not distributed_training or (distributed_training and global_rank == 0):
        log_metrics(confusion_matrix)

        # Print confusion matrix
        print('\nConfusion Matrix:')
        print('----------------')
        print('Predicted →')
        print('Actual ↓')
        print('      ' + ''.join([f'{classes[i]:<7}' for i in range(10)]))
        for i in range(10):
            print(f'{classes[i]:<6}' + ''.join([f'{confusion_matrix[i, j].item():7d}' for j in range(10)]))

        # Log confusion matrix
        logging.info('\nConfusion Matrix:')
        logging.info('----------------')
        logging.info('Predicted ->')
        logging.info('Actual ↓')
        logging.info('      ' + ''.join([f'{classes[i]:<7}' for i in range(10)]))
        for i in range(10):
            logging.info(f'{classes[i]:<6}' + ''.join([f'{confusion_matrix[i, j].item():7d}' for j in range(10)]))
