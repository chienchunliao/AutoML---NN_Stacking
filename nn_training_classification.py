   # -*- coding: utf-8 -*-
import numpy as np, torch, torch.nn as nn
from datetime import datetime
from torcheval.metrics.functional import r2_score

class CustomDataset(torch.utils.data.Dataset):
  def __init__(self, X, y):
    self.X = np.array(X)
    self.y = y
    
  def __len__(self):
    return len(self.X)
    
  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()
    sample = (self.X[idx], self.y[idx])
    return sample

def split_dataset(base_dataset, fraction, seed):
    split_a_size = int(fraction * len(base_dataset))
    split_b_size = len(base_dataset) - split_a_size
    return torch.utils.data.random_split(base_dataset, [split_a_size, split_b_size], generator=torch.Generator().manual_seed(seed)
    )

def init_weights(layer):
  if type(layer) == nn.Linear:
    torch.nn.init.kaiming_normal_(layer.weight)
    # torch.nn.init.normal_(layer.weight, mean = 0, std = 0.001)
    torch.nn.init.zeros_(layer.bias)

def train(train_loader, 
          loss_function, 
          model, optimizer, 
          grad_clipping, 
          max_norm, 
          device, 
          scheduler, 
          scoring='r2'):

  # Training Loop 

  # initilalize variables as global
  # these counts will be updated every epoch

  # Initialize train_loss at the he start of the epoch
  if scoring == 'r2':
      preds = torch.Tensor() 
      preds = preds.to(device)
      y_true = torch.Tensor()
      y_true = y_true.to(device)
  else:
      running_train_loss = 0
      running_train_tp = 0
      running_train_tn = 0
      running_train_fp = 0
      running_train_fn = 0
  
  # put the model in training mode

  model.train()
  
  # Iterate on batches from the dataset using train_loader
  for input_, targets in train_loader:
    
    # move inputs and outputs to GPUs
    input_ = input_.to(device)
    targets = targets.to(device)


    # Step 1: Forward Pass: Compute model's predictions 
    output = model(input_)
    
    # Step 2: Compute loss
    targets = targets.long()
    loss = loss_function(output, targets)

    
    # Step 3: Backward pass -Compute the gradients
    optimizer.zero_grad()
    loss.backward()

    # Gradient Clipping
    if grad_clipping:
      nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm, norm_type=2)

    # Step 4: Update the parameters
    optimizer.step()
    
    if scheduler:
        scheduler.step()
        
    # Step 5: update the variables for score
    if scoring == 'r2':
        preds = torch.cat((preds, output))
        y_true = torch.cat((y_true, targets))
    else:
        y_pred_labels = torch.argmax(output, dim = 1)
        tp = torch.sum((y_pred_labels == targets) & (y_pred_labels==1))
        tn = torch.sum((y_pred_labels == targets) & (y_pred_labels==0))
        fp = torch.sum((y_pred_labels != targets) & (y_pred_labels==1))
        fn = torch.sum((y_pred_labels != targets) & (y_pred_labels==0))
        running_train_tp += tp
        running_train_tn += tn
        running_train_fp += fp
        running_train_fn += fn
          
    # Add train loss of a batch 
    running_train_loss += loss.item()
    
  
  # Calculate mean train loss for the whole dataset for a particular epoch
  train_loss = running_train_loss/len(train_loader)

  # Calculate accuracy for the whole dataset for a particular epoch
  if scoring == 'r2':
      train_score = r2_score(preds, y_true)
  elif scoring == 'accuracy':
      train_score = (running_train_tp+running_train_tn)/len(train_loader.dataset)
  elif scoring == 'recall':
      train_score = running_train_tp / (running_train_tp + running_train_fn)
  elif scoring == 'precision':
      train_score = running_train_tp / (running_train_tp + running_train_fp)
  elif scoring == 'f1':
      train_score = running_train_tp / (running_train_tp + 1/2*(running_train_fp + running_train_fn))
  else:
      raise ValueError("Please use 'r2', 'accuracy', 'precision', 'recall', or 'f1' for scoring.")
  

  return train_loss, train_score


def validate(valid_loader, 
             loss_function, 
             model, 
             device, 
             scoring='r2'):

  # initilalize variables as global
  # these counts will be updated every epoch

  # Validation/Test loop
  # Initialize valid_loss at the start of the epoch
  if scoring == 'r2':
      preds = torch.Tensor() 
      preds = preds.to(device)
      y_true = torch.Tensor()
      y_true = y_true.to(device)
  else:
      running_val_loss = 0
      running_val_tp = 0
      running_val_tn = 0
      running_val_fp = 0
      running_val_fn = 0
  # put the model in evaluation mode
  model.eval()

  with torch.no_grad():
    for input_,targets in valid_loader:

      # move inputs and outputs to GPUs
      input_ = input_.to(device)
      targets = targets.to(device)

      # Step 1: Forward Pass: Compute model's predictions 
      output = model(input_)

      # Step 2: Compute loss
      targets = targets.long()
      loss = loss_function(output, targets)

      # Add val loss of a batch 
      running_val_loss += loss.item()

      # Add correct count for each batch
      if scoring == 'r2':
          preds = torch.cat((preds, output))
          y_true = torch.cat((y_true, targets))
      else:
          y_pred_labels = torch.argmax(output, dim = 1)
          tp = torch.sum((y_pred_labels == targets) & (y_pred_labels==1))
          tn = torch.sum((y_pred_labels == targets) & (y_pred_labels==0))
          fp = torch.sum((y_pred_labels != targets) & (y_pred_labels==1))
          fn = torch.sum((y_pred_labels != targets) & (y_pred_labels==0))
          running_val_tp += tp
          running_val_tn += tn
          running_val_fp += fp
          running_val_fn += fn

    # Calculate mean val loss for the whole dataset for a particular epoch
    val_loss = running_val_loss/len(valid_loader)

    # Calculate accuracy for the whole dataset for a particular epoch
    if scoring == 'r2':
        val_score = r2_score(preds, y_true)
    elif scoring == 'accuracy':
        val_score = (running_val_tp+running_val_tn)/len(valid_loader.dataset)
    elif scoring == 'recall':
        val_score = running_val_tp / (running_val_tp + running_val_fn)
    elif scoring == 'precision':
        val_score = running_val_tp / (running_val_tp + running_val_fp)
    elif scoring == 'f1':
        val_score = running_val_tp / (running_val_tp + 1/2*(running_val_fp + running_val_fn))
    else:
        raise ValueError("Please use 'r2', 'accuracy', 'precision', 'recall', or 'f1' for scoring.")

    
  return val_loss, val_score

def train_loop(train_loader, valid_loader, model, optimizer, loss_function, epochs, device, patience, early_stopping,
               file_model, grad_clipping, max_norm, scheduler, scoring='accuracy'):
    
  """ 
  Function for training the model and plotting the graph for train & validation loss vs epoch.
  Input: iterator for train dataset, initial weights and bias, epochs, learning rate, batch size.
  Output: final weights, bias and train loss and validation loss for each epoch.
  """

  # Create lists to store train and val loss at each epoch
  train_loss_history = []
  valid_loss_history = []
  train_score_history = []
  valid_score_history = []

  # initialize variables for early stopping

  delta = 0
  best_score = None
  valid_loss_min = np.Inf
  counter_early_stop=0
  early_stop=False

  # Iterate for the given number of epochs
  # Step 5: Repeat steps 1 - 4

  for epoch in range(epochs):

    t0 = datetime.now()

    # Get train loss and accuracy for one epoch
    train_loss, train_score = train(train_loader, 
                                    loss_function, 
                                    model, 
                                    optimizer, 
                                    grad_clipping, 
                                    max_norm, 
                                    device,
                                    scheduler,
                                    scoring)
    valid_loss, valid_score   = validate(valid_loader, 
                                         loss_function, 
                                         model, 
                                         device,
                                         scoring)

    dt = datetime.now() - t0

    # Save history of the Losses and accuracy
    train_loss_history.append(train_loss)
    train_score_history.append(train_score)

    valid_loss_history.append(valid_loss)
    valid_score_history.append(valid_score)

    # Log the train and valid loss to wandb

    if early_stopping:
      score = -valid_loss
      if best_score is None:
        best_score=score
        print(f'Validation loss has decreased ({valid_loss_min:.6f} --> {valid_loss:.6f}). Saving Model...')
        torch.save(model.state_dict(), file_model)
        valid_loss_min = valid_loss

      elif score <= best_score + delta:
        counter_early_stop += 1
        print(f'Early stoping counter: {counter_early_stop} out of {patience}')
        if counter_early_stop > patience:
          early_stop = True
      elif score > best_score:
        best_score = score
        print(f'Validation loss has decreased ({valid_loss_min:.6f} --> {valid_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), file_model)
        counter_early_stop=0
        valid_loss_min = valid_loss

      
      else:
        pass
        # best_score = score
        # print(f'Validation loss has decreased ({valid_loss_min:.6f} --> {valid_loss:.6f}). Saving model...')
        # torch.save(model.state_dict(), file_model)
        # counter_early_stop=0
        # valid_loss_min = valid_loss

      if early_stop:
        print('Early Stopping')
        break

    else:

      score = -valid_loss
      if best_score is None:
        best_score=score
        print(f'Validation loss has decreased ({valid_loss_min:.6f} --> {valid_loss:.6f}). Saving Model...')
        torch.save(model.state_dict(), file_model)
        valid_loss_min = valid_loss

      elif score < best_score + delta:
        print(f'Validation loss has not decreased ({valid_loss_min:.6f} --> {valid_loss:.6f}). Not Saving Model...')
      
      else:
        best_score = score
        print(f'Validation loss has decreased ({valid_loss_min:.6f} --> {valid_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), file_model)
        valid_loss_min = valid_loss
    
    # Print the train loss and accuracy for given number of epochs, batch size and number of samples
    print(f'Epoch : {epoch+1} / {epochs}')
    print(f'Time to complete {epoch+1} is {dt}')
    if scheduler:
      print(f'Learning rate: {scheduler._last_lr[0]}')
    print(f'Train Loss: {train_loss : .4f} | Train {scoring}: {train_score : .4f}')
    print(f'Valid Loss: {valid_loss : .4f} | Valid {scoring}: {valid_score : .4f}')
    print()
    torch.cuda.empty_cache()

  return train_loss_history, train_score_history, valid_loss_history, valid_score_history

# def get_acc_pred(data_loader, model, device):
    
#   """ 
#   Function to get predictions and accuracy for a given data using estimated model
#   Input: Data iterator, Final estimated weoights, bias
#   Output: Prections and Accuracy for given dataset
#   """

#   # Array to store predicted labels
#   predictions = torch.Tensor() # empty tensor
#   predictions = predictions.to(device) # move predictions to GPU

#   # Array to store actual labels
#   y = torch.Tensor() # empty tensor
#   y = y.to(device)

#   # put the model in evaluation mode
#   model.eval()
  
#   # Iterate over batches from data iterator
#   with torch.no_grad():
#     for input_, targets in data_loader:
      
#       # move inputs and outputs to GPUs
      
#       input_ = input_.to(device)
#       targets = targets.to(device)
      
#       # Calculated the predicted labels
#       output = model(input_)

#       # Choose the label with maximum probability
#       prediction = torch.argmax(output, dim = 1)

#       # Add the predicted labels to the array
#       predictions = torch.cat((predictions, prediction)) 

#       # Add the actual labels to the array
#       y = torch.cat((y, targets)) 

#   # Check for complete dataset if actual and predicted labels are same or not
#   # Calculate accuracy
#   acc = (predictions == y).float().mean()

#   # Return tuple containing predictions and accuracy
#   return predictions, acc 

# def get_pred_prob(data_loader, model, device):
    
#   """ 
#   Function to get predictions and accuracy for a given data using estimated model
#   Input: Data iterator, Final estimated weoights, bias
#   Output: Prections and Accuracy for given dataset
#   """

#   # Array to store predicted labels
#   predictions = torch.Tensor() # empty tensor
#   predictions = predictions.to(device) # move predictions to GPU

#   # Array to store actual labels
#   #y = torch.Tensor() # empty tensor
#   #y = y.to(device)

#   # put the model in evaluation mode
#   model.eval()
  
#   # Iterate over batches from data iterator
#   with torch.no_grad():
#     for input_, targets in data_loader:
      
#       # move inputs and outputs to GPUs
      
#       input_ = input_.to(device)
#       #targets = targets.to(device)
      
#       # Calculated the predicted labels
#       output = model(input_)

#       # Choose the label with maximum probability
#       prediction = output[:,1]
#       #prediction = torch.argmax(output, dim = 1)

#       # Add the predicted labels to the array
#       predictions = torch.cat((predictions, prediction)) 

#       # Add the actual labels to the array
#       #y = torch.cat((y, targets)) 

#   # Check for complete dataset if actual and predicted labels are same or not
#   # Calculate accuracy
#   #acc = (predictions == y).float().mean()

#   # Return tuple containing predictions and accuracy
#   return predictions

# def get_pred(data_loader, model, device):
    
#   """ 
#   Function to get predictions and accuracy for a given data using estimated model
#   Input: Data iterator, Final estimated weoights, bias
#   Output: Prections and Accuracy for given dataset
#   """

#   # Array to store predicted labels
#   predictions = torch.Tensor() # empty tensor
#   predictions = predictions.to(device) # move predictions to GPU

#   # Array to store actual labels
#   #y = torch.Tensor() # empty tensor
#   #y = y.to(device)

#   # put the model in evaluation mode
#   model.eval()
  
#   # Iterate over batches from data iterator
#   with torch.no_grad():
#     for input_, targets in data_loader:
      
#       # move inputs and outputs to GPUs
      
#       input_ = input_.to(device)
#       #targets = targets.to(device)
      
#       # Calculated the predicted labels
#       output = model(input_)

#       # Choose the label with maximum probability
#       #prediction = output[:,0]
#       prediction = torch.argmax(output, dim = 1)

#       # Add the predicted labels to the array
#       predictions = torch.cat((predictions, prediction)) 

#       # Add the actual labels to the array
#       #y = torch.cat((y, targets)) 

#   # Check for complete dataset if actual and predicted labels are same or not
#   # Calculate accuracy
#   #acc = (predictions == y).float().mean()

#   # Return tuple containing predictions and accuracy
#   return predictions


def get_pred(data_loader, 
             model, 
             device,  
             return_ytrue=False,
             return_proba=False,
             ):
    
  """ 
  Function to get prediction or true y a given data using estimated model
  Input: data_loader: Data iterator
         model: trained model
         device: device for model to load on
         return_ytrue: return true y or not
         return_proba: for regression, always setting to True
                       for classification, True for getting the probability; 
                           False for getting the labels 
  Output: Prections (,y_true(optional))
  """

  # Array to store predicted labels
  preds = torch.Tensor() # empty tensor
  preds = preds.to(device) # move predictions to GPU

  # Array to store actual labels and predicted probability
  if return_ytrue:
      y_true = torch.Tensor()
      y_true = y_true.to(device)
  else:
      y_true = None

  # put the model in evaluation mode
  model.eval()
  
  # Iterate over batches from data iterator
  with torch.no_grad():
    for input_, targets in data_loader:
      
      # move inputs and outputs to GPUs
      
      input_ = input_.to(device)
      targets = targets.to(device)
      
      # Calculated the predicted labels
      output = model(input_)

      # Choose the label with maximum probability
      #prediction = output[:,0]
      if not return_proba:
          output = torch.argmax(output, dim = 1)

      # Add the predicted labels to the array
      preds = torch.cat((preds, output))

      if return_ytrue:
          y_true = torch.cat((y_true, targets))
  
  return preds, y_true
