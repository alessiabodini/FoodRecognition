import torch
import numpy as np
from torch import nn
from matplotlib import pyplot as plt

def nn(train_set, test_set, train_labels, test_labels):

    # 1. Set parameters
    dtype = torch.float
    device = torch.device('cuda')
    learning_rate = 1e-6
    N, D_in, H, D_out = train_set.shape[0] + test_set.shape[0], 2, 2, 1

    # 2. Create a model
    model = nn.Sequential(nn.Linear(D_in, H), nn.ReLU(), nn.Linear(H, D_out))

    # 3. Calculate the loss function
    loss_fn = nn.MSELoss(reduction='sum')

    # 4. Define the optimization function
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(1000): # 10000
        pred_train = model(train_set)
        loss_train = loss_fn(pred_train.squeeze(), train_labels)

    with torch.no_grad():
        pred_test = model(test_set)
    
    if epoch % 250 == 0:
        print(epoch, loss_train.item())
        plt.scatter(test_set[:,0].squeeze().detach().numpy(), pred_test.squeeze().detach().numpy(), c='b')
        plt.scatter(test_set[:,0].squeeze().detach().numpy(), test_labels.squeeze().detach().numpy(), c='r')
        plt.show()

    optimizer.zero_grad()
    loss_train.backward()
    optimizer.step()