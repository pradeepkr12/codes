import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import torch.nn.functional as F

class SimpleClassifier(nn.Module):
    def __init__(self, num_inputs, num_hidden,
                 num_outputs):
        super(SimpleClassifier, self).__init__()
        self.linear = nn.Linear(num_inputs, num_hidden)
        self.act_fn = nn.Tanh()
        self.linear2 = nn.Linear(num_hidden, num_outputs)

    def forward(self, x):
        x = self.linear(x)
        x = self.act_fn(x)
        x = self.linear2(x)
        return x

model = SimpleClassifier(num_inputs=2,
                         num_hidden=4,
                         num_outputs=1)

for name, param in model.named_parameters():
    print (f"{name}, {param.shape}")

import torch.utils.data as data

class XORDataset(data.Dataset):
    def __init__(self, size, std=0.1):
        super(XORDataset, self).__init__()
        self.size = size
        self.std = std
        self.generate_continuous_xor()

    def generate_continuous_xor(self):
        data = torch.randint(low=0, high=2,
                             size=(self.size, 2),
                             dtype=torch.float32)
        label = (data.sum(dim=1) == 1).to(torch.long)
        data += self.std * torch.randn(data.shape)

        self.data = data
        self.label = label
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        data_point = self.data[idx]
        data_label = self.label[idx]
        return data_point, data_label

dataset = XORDataset(size=200)
print ("Size:", len(dataset))
print ("Datapoint ", dataset[0])

def visualize_samples(data, label):
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    if isinstance(label, torch.Tensor):
        label = label.cpu().numpy()

    data_0 = data[label == 0]
    data_1 = data[label == 1]

    plt.figure(figsize=(4,4))
    plt.scatter(data_0[:, 0], data_0[:, 1],
                edgecolor="#333", label="Class 0")
    plt.scatter(data_1[:, 0], data_1[:, 1],
                edgecolor="#333", label="Class 1")
    plt.title("Dataset Samples")
    plt.ylabel("x2")
    plt.xlabel("x1")
    plt.legend()

visualize_samples(dataset.data, dataset.label)
plt.show()

data_loader = data.DataLoader(dataset,
                              batch_size=8,
                              shuffle=True)

data_inputs, data_labels = next(iter(data_loader))

# optimization

loss_module = nn.BCEWithLogitsLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

train_dataset = XORDataset(size=2500)
train_data_loader = data.DataLoader(train_dataset,
                                    batch_size=128,
                                    shuffle=True)

device = torch.device('cpu')
model.to(device)

def train_model(model, optimizer, data_loader,
                loss_module, num_epochs=100):
    model.train()
    for epoch in tqdm(range(num_epochs)):
        for data_inputs, data_labels in data_loader:
            # step 1
            data_inputs = data_inputs.to(device)
            data_labels = data_labels.to(device)

            preds = model(data_inputs)
            preds = preds.squeeze(dim=1)

            # step 2
            loss = loss_module(preds, data_labels.float())

            # step 3
            optimizer.zero_grad()
            loss.backward()

            # step 4
            optimizer.step()


train_model(model, optimizer, train_data_loader, loss_module)

# save the model

state_dict = model.state_dict()
torch.save(state_dict, "model.tar")

# load model
state_dict = torch.load("model.tar")

new_model = SimpleClassifier(num_inputs=2,
                             num_hidden=4,
                             num_outputs=1)
new_model.load_state_dict(state_dict)

# model evaluation

def eval_model(model, data_loader):
    model.eval()
    true_preds, num_preds = 0., 0.
    with torch.no_grad():
        for data_inputs, data_labels in data_loader:
            data_inputs = data_inputs.to(device)
            data_labels = data_labels.to(device)
            preds = model(data_inputs)
            preds = preds.squeeze(dim=1)
            preds = torch.sigmoid(preds)
            pred_labels = (preds >= 0.5).long()
            true_preds += (pred_labels == data_labels).sum()
            num_preds += data_labels.shape[0]

    acc = true_preds/num_preds
    print (f"Accuracy of model: ", {100.0*acc: 4.2f})

eval_model(model, test_data_loader)

# tensorboard logging
from torch.utils.tensorboard import SummaryWriter

# loading tensorboard in notebook
%load_ext tensorboard

'''
usage
writer = SummaryWriter()
writer.add_scaler
writer.add_graph
'''
def train_model_with_logger(model, optimizer, data_loader, loss_module, val_dataset, num_epochs=100, logging_dir='runs/our_experiment'):
    # Create TensorBoard logger
    writer = SummaryWriter(logging_dir)
    model_plotted = False

    # Set model to train mode
    model.train()

    # Training loop
    for epoch in tqdm(range(num_epochs)):
        epoch_loss = 0.0
        for data_inputs, data_labels in data_loader:

            ## Step 1: Move input data to device (only strictly necessary if we use GPU)
            data_inputs = data_inputs.to(device)
            data_labels = data_labels.to(device)

            # For the very first batch, we visualize the computation graph in TensorBoard
            if not model_plotted:
                writer.add_graph(model, data_inputs)
                model_plotted = True

            ## Step 2: Run the model on the input data
            preds = model(data_inputs)
            preds = preds.squeeze(dim=1) # Output is [Batch size, 1], but we want [Batch size]

            ## Step 3: Calculate the loss
            loss = loss_module(preds, data_labels.float())

            ## Step 4: Perform backpropagation
            # Before calculating the gradients, we need to ensure that they are all zero.
            # The gradients would not be overwritten, but actually added to the existing ones.
            optimizer.zero_grad()
            # Perform backpropagation
            loss.backward()

            ## Step 5: Update the parameters
            optimizer.step()

            ## Step 6: Take the running average of the loss
            epoch_loss += loss.item()

        # Add average loss to TensorBoard
        epoch_loss /= len(data_loader)
        writer.add_scalar('training_loss',
                          epoch_loss,
                          global_step = epoch + 1)

        # Visualize prediction and add figure to TensorBoard
        # Since matplotlib figures can be slow in rendering, we only do it every 10th epoch
        if (epoch + 1) % 10 == 0:
            fig = visualize_classification(model, val_dataset.data, val_dataset.label)
            writer.add_figure('predictions',
                              fig,
                              global_step = epoch + 1)

    writer.close()

# to start tensorboard
%tensorboard --logdir runs/our_experiment

