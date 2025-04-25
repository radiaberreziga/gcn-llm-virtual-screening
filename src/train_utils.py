import torch
import numpy as np
from sklearn.metrics import f1_score 

def train(train_loader,model,device, criterion, optimizer):
    model.train()
    all_embeddings = []  # Store embeddings for visualization

    for data in train_loader:  # Iterate in batches over the training dataset.

         data = data.to(device)  # Move data to GPU
         out  = model(data.x.float(), data.edge_index, data.batch, data.smile_mol)  # Perform a single forward pass. for GCN-LLM
         #out  = model(data.x.float(), data.edge_index, data.batch)  # Perform a single forward pass. for GCN

         loss = criterion(out, data.y)  # Compute the loss.
         loss.backward()  # Derive gradients.
         optimizer.step()  # Update parameters based on gradients.
         optimizer.zero_grad()  # Clear gradients.


def test(loader,model,device):
     model.eval()

     correct = 0
     all_preds = []
     all_labels = []
     all_probs = []

     for data in loader:  # Iterate in batches over the training/test dataset.

         data = data.to(device)  # Move data to GPU
         out  = model(data.x.float(), data.edge_index, data.batch, data.smile_mol)  # Perform a single forward pass. for GCN-LLM
         #out  = model(data.x.float(), data.edge_index, data.batch)  # Perform a single forward pass. for GCN

         pred = out.argmax(dim=1)  # Use the class with highest probability.
         probs = torch.nn.functional.softmax(out, dim=1)[:, 1]  # Get probabilities for the positive class

         correct += int((pred == data.y).sum())  # Check against ground-truth labels.
         all_preds.append(pred.cpu().numpy())
         all_labels.append(data.y.cpu().numpy())
         all_probs.append(probs.cpu().detach().numpy())

     all_preds = np.concatenate(all_preds)
     all_labels = np.concatenate(all_labels)
     all_probs = np.concatenate(all_probs)

     acc = correct / len(loader.dataset)


     f1_val  = f1_score(all_labels, all_preds, average='binary')

     return acc, f1_val , all_labels, all_probs