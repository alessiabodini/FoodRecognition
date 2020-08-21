import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import os
import numpy as np
from pathlib import Path
from tools import loadImages
import matplotlib.pyplot as plt

path = Path(os.path.join('C:/', 'Users', 'ale19', 'Downloads', 'Food-101'))
path_h5 = path
n_feat = 2048
n_classes = 101
learning_rate = 1e-6
batch_size = 4
epochs = 2


def show(images):
    fig = plt.figure(figsize=(8, 8))
    for i in range(4):
        fig.add_subplot(1, 4, i+1)
        npimg = images[i].numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    #plt.pause(1)
    plt.close('all')


def nnet_feat(train_set, test_set, train_labels, test_labels, classes):

    # 1. Convert features set and labels in tensors
    train_set =  torch.tensor(train_set, dtype=torch.float32)
    test_set = torch.tensor(test_set, dtype=torch.float32)
    print('Dataset loaded.')
                    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    train_labels = torch.tensor(train_labels, dtype=torch.long)
    test_labels = torch.tensor(test_labels, dtype=torch.long)

    # 2. Show some images
    data_iter = iter(train_loader)
    images = data_iter.next()
    #show(images)
    #print(' '.join('%5s' % classes[int(train_labels[i])] for i in range(4)))

    # 3. Create a model
    net = nn.Sequential(nn.Linear(n_feat, 512), nn.ReLU(), nn.Linear(512, n_classes))

    # 4. Define a loss function and the optimizer
    criterion = nn.CrossEntropyLoss() # or MSELoss 
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
    #optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # 5. Train the network (if necessary)
    if os.path.isfile('trained_network_epoch10.pth'):
        net.load_state_dict(torch.load('trained_network_epoch10.pth'))
    
    else:
        for epoch in range(epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, inputs in enumerate(train_loader):
                curr_size = inputs.size()[0]
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, train_labels[i*4:(i+1)*4])
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Print statistics
                running_loss += loss.item()
                if i % 500 == 0:
                    print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

        print('Training ended.')

        # Save the obtained model
        torch.save(net.state_dict(), 'trained_network.pth')

    # 6. Test the network
    dataiter = iter(test_loader)
    images = dataiter.next()
    #show(images)
    print('GroundTruth: ', ' '.join('%5s' % classes[int(test_labels[i])] for i in range(4)))

    #torch.transpose(images, 1, 3)
    outputs = net(images)
    predicted = torch.argmax(outputs, dim=1)
    print('Predicted: ', ' '.join('%5s' % classes[int(predicted[i])] for i in range(4)))

    correct = 0
    total = 0
    with torch.no_grad():
        for i, inputs in enumerate(test_loader):
            curr_size = inputs.size()[0]
            outputs = net(inputs)
            predicted = torch.argmax(outputs, dim=1)
            total += curr_size
            correct += (predicted == test_labels[i*4:(i+1)*4]).sum().item()
            if i % 50 == 0:
                print(predicted)
                print(test_labels[i*4:(i+1)*4], end='\n')

    print('Accuracy of the network on the 1000 test images: %d %%' % (100 * correct/total))

    # 7. Test performance for every class
    class_correct = list(0. for i in range(n_classes))
    class_total = list(0. for i in range(n_classes))
    with torch.no_grad():
        for i, inputs in enumerate(test_loader):
            curr_size = inputs.size()[0]
            outputs = net(inputs)
            predicted = torch.argmax(outputs, dim=1)
            c = (predicted == test_labels[i*4:(i+1)*4]).squeeze()
            for j in range(curr_size):
                label = test_labels[i*batch_size+j]
                class_correct[int(label)] += c[j].item()
                class_total[int(label)] += 1

    for i in range(n_classes):
        print('Accuracy of %5s: %2d %%' % (classes[i], 100 * class_correct[i]/class_total[i]))
