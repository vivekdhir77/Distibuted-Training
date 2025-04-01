import warnings
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.optim import AdamW
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import init_process_group, destroy_process_group
import os
from tqdm import tqdm
from vit import VisionTransformer
from data_loader import get_data_loaders, trainset, testset, classes
from config import get_default_config, get_weights_file_path, get_latest_weights_file_path, ModelConfig
import logging
from datetime import datetime
import time

def evaluate_model(model, test_loader, device, num_classes=10):
    """Evaluate model performance and return confusion matrix"""
    model.eval()
    confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    
    with torch.no_grad():
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

def log_metrics(confusion_matrix, classes):
    """Calculate and log metrics for each class based on confusion matrix"""
    print('\nMetrics for each class:')
    print('------------------------')
    
    for i in range(len(classes)):
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
    
    # Calculate overall accuracy
    total_correct = confusion_matrix.diag().sum().item()
    total_samples = confusion_matrix.sum().item()
    overall_accuracy = total_correct / total_samples
    
    print(f'\nOverall Accuracy: {overall_accuracy:.4f}')
    logging.info(f'Overall Accuracy: {overall_accuracy:.4f}')
    
    return overall_accuracy

def train_model(config: ModelConfig, distributed_training=False):
  assert torch.cuda.is_available(), "Training on CPU is not supported"
  
  # Set device appropriately
  if distributed_training:
      device = torch.device(f'cuda:{config.local_rank}')
      print(f"GPU {config.local_rank} - Using device: {device} (distributed mode)")
  else:
      device = torch.device('cuda:0')
      print(f"Using device: {device} (single GPU mode)")
  
  # Get data loaders
  trainloader, testloader = get_data_loaders(distributed_training, config.local_rank)
  
  # Set up logging
  os.makedirs(config.model_folder, exist_ok=True)
  logging.basicConfig(
      filename=f'vit_training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
      level=logging.INFO,
      format='%(asctime)s - %(message)s'
  )
  
  learning_rate = config.learning_rate
  num_epochs = config.num_epochs
  
  model = VisionTransformer(
      n_channels=3, embed_dim=config.embed_dim, num_layers=config.num_layers,
      num_heads=config.num_heads, forward_expansion=config.forward_expansion,
      image_size=config.image_size, patch_size=config.patch_size,
      num_classes=config.num_classes, dropout=config.dropout, stochastic_depth=config.stochastic_depth
  )

  model.to(device)

  criterion = nn.CrossEntropyLoss()
  optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
  scaler = torch.cuda.amp.GradScaler()
  initial_epoch = 0
  global_step = 0
  
  # Track metrics history
  train_losses = []
  train_accuracies = []
  val_accuracies = []
  best_accuracy = 0.0
  total_train_time = 0  # Initialize total training time
  
  # Load checkpoint if exists
  if config.preload != '':
    if config.preload == 'latest':
      model_filename = get_latest_weights_file_path(config)
    else:
      model_filename = get_weights_file_path(config, int(config.preload))

    if model_filename is not None:
      print(f"Preloading model {model_filename}")
      try:
          # Load checkpoint with appropriate device mapping
          if distributed_training:
              state = torch.load(model_filename, map_location=f'cuda:{config.local_rank}')
          else:
              state = torch.load(model_filename, map_location='cuda:0')
              
          model.load_state_dict(state['model_state_dict'])
          initial_epoch = state['epoch'] + 1  # Start from next epoch
          optimizer.load_state_dict(state['optimizer_state_dict'])
          global_step = state.get('global_step', 0)
          
          # Load accuracy histories if available
          if 'train_accuracies_history' in state:
              train_accuracies = state['train_accuracies_history']
              print(f"Loaded {len(train_accuracies)} previous train accuracy records")
          
          if 'test_accuracies_history' in state:
              val_accuracies = state['test_accuracies_history']
              print(f"Loaded {len(val_accuracies)} previous test accuracy records")
              
          if 'best_accuracy' in state:
              best_accuracy = state['best_accuracy']
              print(f"Previous best accuracy: {best_accuracy:.4f}")
          
          # Load total training time if available
          if 'total_train_time' in state:
              total_train_time = state['total_train_time']
              print(f"Loaded previous training time: {total_train_time:.2f} seconds")
          
          print(f"Continuing training from epoch {initial_epoch}")
          del state
      except Exception as e:
          print(f"Error loading checkpoint: {e}")
          print("Starting training from scratch")
    else:
      print(f"Could not find model to preload: {config.preload}. Starting from scratch")
  
  # Wrap model in DistributedDataParallel only if in distributed mode
  if distributed_training:
      model = DistributedDataParallel(model, device_ids=[config.local_rank])
  
  # Training loop
  total_train_time = 0
  for epoch in range(initial_epoch, num_epochs):
    epoch_start_time = time.time()
    torch.cuda.empty_cache()
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Set epoch for DistributedSampler (only in distributed mode)
    if distributed_training and hasattr(trainloader, 'sampler') and isinstance(trainloader.sampler, DistributedSampler):
        trainloader.sampler.set_epoch(epoch)
    
    for batch_idx, (images, labels) in enumerate(trainloader):
          images, labels = images.to(device), labels.to(device)

          optimizer.zero_grad()
          with torch.cuda.amp.autocast():  # Use mixed precision
              # Forward pass - handle both distributed and non-distributed cases
              if isinstance(model, DistributedDataParallel):
                  outputs = model(images)
              else:
                  outputs = model(images)
                  
              loss = criterion(outputs, labels)

          scaler.scale(loss).backward()
          scaler.step(optimizer)
          scaler.update()

          running_loss += loss.item()
          _, predicted = torch.max(outputs, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()
          
          # Log progress every 100 batches
          if (batch_idx + 1) % 100 == 0:
              batch_loss = loss.item()
              batch_acc = 100 * (predicted == labels).sum().item() / labels.size(0)
              print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(trainloader)}], "
                    f"Loss: {batch_loss:.4f}, Acc: {batch_acc:.2f}%")
    
    # Calculate epoch statistics
    epoch_loss = running_loss / len(trainloader)
    epoch_acc = 100 * correct / total
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)
    
    epoch_time = time.time() - epoch_start_time
    total_train_time += epoch_time
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%, Time: {epoch_time:.2f}s")
    logging.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%, Time: {epoch_time:.2f}s")

    # Evaluate model and save checkpoints
    # Only main process in distributed mode or single GPU mode
    should_save = (not distributed_training) or (distributed_training and config.global_rank == 0)
    
    if should_save:
        print("Evaluating model on test set...")
        confusion_matrix = evaluate_model(model, testloader, device, config.num_classes)
        val_accuracy = log_metrics(confusion_matrix, classes)
        val_accuracies.append(val_accuracy)
        
        # Get the correct state dict depending on model type
        if isinstance(model, DistributedDataParallel):
            model_state_dict = model.module.state_dict()
        else:
            model_state_dict = model.state_dict()
            
        # Save checkpoint after each epoch
        model_filename = get_weights_file_path(config, epoch)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
            'train_accuracy': epoch_acc,
            'test_accuracy': val_accuracy,
            'train_accuracies_history': train_accuracies,
            'test_accuracies_history': val_accuracies,
            'global_step': global_step,
            'total_train_time': total_train_time,
        }, model_filename)
        
        # Also save as latest.pt for easy resuming
        latest_filename = os.path.join(config.model_folder, 'latest.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
            'train_accuracy': epoch_acc,
            'test_accuracy': val_accuracy,
            'train_accuracies_history': train_accuracies,
            'test_accuracies_history': val_accuracies,
            'global_step': global_step,
            'total_train_time': total_train_time,
        }, latest_filename)
        
        print(f"Saved model checkpoint to {model_filename}")
        logging.info(f"Saved model checkpoint to {model_filename}")
        
        # Save best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_filename = os.path.join(config.model_folder, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
                'train_accuracy': epoch_acc,
                'test_accuracy': val_accuracy,
                'train_accuracies_history': train_accuracies,
                'test_accuracies_history': val_accuracies,
                'best_accuracy': best_accuracy,
                'global_step': global_step,
                'total_train_time': total_train_time,
            }, best_filename)
            print(f"Saved best model with accuracy {val_accuracy:.4f}")
            logging.info(f"Saved best model with accuracy {val_accuracy:.4f}")

  # Plot training metrics at the end (only for main process)
  if should_save and len(train_losses) > 0:
      plt.figure(figsize=(12, 5))
      
      plt.subplot(1, 2, 1)
      plt.plot(train_losses)
      plt.title(f'Training Loss\nTotal Train Time: {total_train_time:.2f}s')
      plt.xlabel('Epoch')
      plt.ylabel('Loss')
      
      plt.subplot(1, 2, 2)
      plt.plot(train_accuracies, label='Train')
      plt.plot(val_accuracies, label='Validation')
      plt.title('Accuracy')
      plt.xlabel('Epoch')
      plt.ylabel('Accuracy (%)')
      plt.legend()
      
      plt.tight_layout()
      plt.savefig(os.path.join(config.model_folder, 'training_metrics.png'))
      plt.close()
      
      print(f"Training completed. Best validation accuracy: {best_accuracy:.4f}")
      print(f"Total training time (excluding validation): {total_train_time:.2f} seconds")
      logging.info(f"Training completed. Best validation accuracy: {best_accuracy:.4f}")
      logging.info(f"Total training time (excluding validation): {total_train_time:.2f} seconds")
      
      # Print final confusion matrix
      print("\nFinal Confusion Matrix:")
      print("----------------------")
      confusion_matrix = evaluate_model(model, testloader, device, config.num_classes)
      print('Predicted →')
      print('Actual ↓')
      print('      ' + ''.join([f'{classes[i]:<7}' for i in range(config.num_classes)]))
      for i in range(config.num_classes):
          print(f'{classes[i]:<6}' + ''.join([f'{confusion_matrix[i, j].item():7d}' for j in range(config.num_classes)]))
      
  return total_train_time


if __name__ == '__main__':
  warnings.filterwarnings("ignore")
  config = get_default_config()
  
  # Get environment variables for distributed training
  config.local_rank = int(os.environ.get('LOCAL_RANK', -1))
  config.global_rank = int(os.environ.get('RANK', -1))
  
  # Determine if we're using distributed training
  distributed_training = config.local_rank != -1 and config.global_rank != -1
  
  # Set up distributed training if needed
  if distributed_training:
      print(f"Initializing distributed process group (rank: {config.global_rank}, local_rank: {config.local_rank})")
      init_process_group(backend='nccl')
      torch.cuda.set_device(config.local_rank)
  else:
      print("Running in single-GPU mode")
      # For single GPU training, set to device 0
      config.local_rank = 0
      config.global_rank = 0
      torch.cuda.set_device(0)
      
  # Create weights directory
  os.makedirs(config.model_folder, exist_ok=True)

  # Start training
  start_time = time.time()
  total_train_time = train_model(config, distributed_training)
  total_training_time = time.time() - start_time
  
  print(f"Total training time: {total_training_time:.2f} seconds")
  logging.info(f"Total training time: {total_training_time:.2f} seconds")
  print(f"Total training time (excluding data loading and validation): {total_train_time:.2f} seconds")
  
  # Clean up distributed process group if needed
  if distributed_training:
      destroy_process_group()
      print("Destroyed process group")
