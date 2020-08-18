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
n_classes = 101
batch_size = 4

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 14 * 14, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 101)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x.float())))
        x = self.pool(F.relu(self.conv2(x.float())))
        #print(x.size())
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def show(images):
    fig = plt.figure(figsize=(8, 8))
    for i in range(4):
        fig.add_subplot(1, 4, i+1)
        npimg = images[i].numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    #plt.pause(1)
    plt.close('all')


def nnet(train_labels, test_labels, classes):

    # 1. Import train and test images
    train_file = os.path.join(path_h5, 'food_c101_n10099_r64x64x3.h5')
    test_file = os.path.join(path_h5, 'food_test_c101_n1000_r64x64x3.h5')
    
    train_set, _ = loadImages(train_file)
    test_set, _ = loadImages(test_file)
    train_set = np.transpose(train_set, (0, 3, 1, 2))
    test_set = np.transpose(test_set, (0, 3, 1, 2))
    print('Dataset loaded.')

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    transform = torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    train_labels = torch.tensor(train_labels, dtype=torch.float32)
    test_labels = torch.tensor(test_labels, dtype=torch.float32)

    # 2. Show some images
    data_iter = iter(train_loader)
    images = data_iter.next()
    #show(images)
    #print(' '.join('%5s' % classes[int(train_labels[i])] for i in range(4)))

    # 3. Create a model
    net = Net()

    # 4. Define a loss function and the optimizer
    criterion = nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

    # 5. Train the network
    for epoch in range(2):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, inputs in enumerate(train_loader):
            optimizer.zero_grad()
            for j in range(inputs.size()[0]):
                inputs[j] = transform(inputs[j])
            outputs = net(inputs)
            #predicted = torch.argmax(outputs, dim=1)
            curr_size = outputs.size()[0]
            right_prob = torch.zeros([curr_size, n_classes], dtype=torch.float32)
            for j in range(curr_size):
                right_prob[j, int(train_labels[i*batch_size+j])] = 1
            loss = criterion(outputs, right_prob)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if i % 500 == 0:
                print(outputs)
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
            if i == 2:
                print(outputs)

    print('Training ended.')

    # Save the obtained model
    torch.save(net.state_dict(), os.path.join(path, 'trained_network.pth'))

    # 6. Test the network
    dataiter = iter(test_loader)
    images = dataiter.next()
    show(images)
    print('GroundTruth: ', ' '.join('%5s' % classes[int(test_labels[i])] for i in range(4)))

    #torch.transpose(images, 1, 3)
    outputs = net(images)
    _, predicted = torch.argmax(outputs, dim=1)
    print('Predicted: ', ' '.join('%5s' % classes[int(predicted[i])] for i in range(4)))

    correct = 0
    total = 0
    with torch.no_grad():
        for images in test_loader:
            outputs = net(images)
            _, predicted = torch.argmax(outputs, dim=1)
            total += test_labels.size(0)
            correct += (int(predicted) == int(test_labels)).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct/total))

    # 7. Test performance for every class
    class_correct = list(0. for i in range(n_classes))
    class_total = list(0. for i in range(n_classes))
    with torch.no_grad():
        for images in test_loader:
            outputs = net(images)
            _, predicted = torch.argmax(outputs, dim=1)
            c = (int(predicted) == int(test_labels)).squeeze()
            for i in range(batch_size):
                label = test_labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(n_classes):
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i]/class_total[i]))
