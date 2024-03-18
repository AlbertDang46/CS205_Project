import os
import math
import random

import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
from torchvision.io import read_image, ImageReadMode
from torch.utils.data import DataLoader


dataset_path = "DATASET_PATH"
dataset_class_paths = ["Non_Demented/", "Very_Mild_Demented/", "Mild_Demented/", "Moderate_Demented/"]


class Conv_Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Flatten(start_dim=1, end_dim=-1)
        )

        self.fc_net_1 = nn.Sequential(
            nn.Linear(in_features=25088, out_features=128),
            nn.ReLU()
        )

        self.fc_net_2 = nn.Sequential(
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=4),
            nn.Softmax(dim=-1)
        )

    def forward(self, input):
        output = self.conv_net(input)
        output = self.fc_net_1(output)
        output = self.fc_net_2(output)
        return output
    

class Dataset(torch.utils.data.Dataset):

    def __init__(self, list_IDs, labels):
        'Initialization'
        self.list_IDs = list_IDs
        self.labels = labels
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        X = read_image(path=dataset_path + ID, mode=ImageReadMode.RGB) / 255.0
        y = self.labels[ID]

        return X, y


#### Import training dataset

partition = {
    'train': [],
    'test': []
}

labels = {}

for index, class_path in enumerate(dataset_class_paths):
    image_names = os.listdir(dataset_path + class_path)

    for img_name in image_names:
        sample_id = class_path + img_name
        partition['train'].append(sample_id)
        labels[sample_id] = index

training_set = Dataset(partition['train'], labels)
training_generator = DataLoader(dataset=training_set, batch_size=32, shuffle=True)

#### Create CNN model

def init_weights(nn_module):
    if isinstance(nn_module, nn.Conv2d) or isinstance(nn_module, nn.Linear):
        nn.init.kaiming_normal_(tensor=nn_module.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(nn_module.bias)

conv_model = Conv_Model()
conv_model.conv_net.apply(init_weights)
conv_model.fc_net_1.apply(init_weights)

optimizer = torch.optim.Adam(conv_model.parameters())

#### Get CNN output and target classes

MAX_EPOCHS = 100
iters, losses = [], []

# More weight is given to classes with fewer samples
labels_list = list(labels.values())
class_weights = torch.from_numpy(compute_class_weight(class_weight='balanced', classes=np.unique(labels_list), y=labels_list).astype(np.float32))
loss_function = nn.CrossEntropyLoss(weight=class_weights)

for i in range(MAX_EPOCHS):
    epoch_loss = 0
    batch_num = 0

    for batch, labels in training_generator:
        outputs = conv_model(batch)

#### Cross entropy loss

        loss = loss_function(outputs, labels)
        epoch_loss += loss

#### Optimize model step

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

#### Append training metadata        

        print("EPOCH " + str(i) + " | Batch " + str(batch_num) + " | Loss: " + str(loss))
        batch_num += 1
    
    iters.append(i)
    losses.append(epoch_loss)
        
#### Plot data
    
plt.title("Learning Curve")
plt.plot(iters, losses, label="Train")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.show()