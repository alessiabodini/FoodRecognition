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
learning_rate = 1e-3
batch_size = 4
epochs = 10

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 14 * 14, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, n_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x.float())))
        x = self.pool(F.relu(self.conv2(x.float())))
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
    train_set = train_set.astype('float32')
    test_set = test_set.astype('float32')
    print('Dataset loaded.')
                    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    train_labels = torch.tensor(train_labels, dtype=torch.long)
    test_labels = torch.tensor(test_labels, dtype=torch.long)

    transform = torchvision.transforms.Normalize((128, 128, 128), (127, 127, 127))

    # 2. Show some images
    data_iter = iter(train_loader)
    images = data_iter.next()
    for i in range(batch_size):
        images[i] = transform(images[i])
    #show(images)
    #print(' '.join('%5s' % classes[int(train_labels[i])] for i in range(4)))

    # 3. Create a model
    net = Net()

    # 4. Define a loss function and the optimizer
    criterion = nn.CrossEntropyLoss() # or MSELoss 
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    #optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # 5. Train the network (if necessary)
    net.train()
    if os.path.isfile('trained_network_epoch.pth'):
        net.load_state_dict(torch.load('trained_network_epoch.pth'))
    
    else:
        for epoch in range(epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, inputs in enumerate(train_loader):
                curr_size = inputs.size()[0]
                optimizer.zero_grad()
                for j in range(curr_size):
                    inputs[j] = transform(inputs[j])
                outputs = net(inputs)
                #right_prob = torch.zeros([curr_size, n_classes], dtype=torch.float32)
                #for j in range(curr_size):
                #    right_prob[j, int(train_labels[i*batch_size+j])] = 1
                loss = criterion(outputs, train_labels[i*batch_size:(i+1)*batch_size])
                
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
    net.eval() # disable batch normalization
    dataiter = iter(test_loader)
    images = dataiter.next()
    for i in range(batch_size):
        images[i] = transform(images[i])
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
            for j in range(curr_size):
                inputs[j] = transform(inputs[j])
            outputs = net(inputs)
            predicted = torch.argmax(outputs, dim=1)
            total += curr_size
            correct += (predicted == test_labels[i*batch_size:(i+1)*batch_size]).sum().item()
            if i % 50 == 0:
                print(predicted)
                print(test_labels[i*batch_size:(i+1)*batch_size], end='\n')

    print('Accuracy of the network on the 1000 test images: %d %%' % (100 * correct/total))

    # 7. Test performance for every class
    class_correct = list(0. for i in range(n_classes))
    class_total = list(0. for i in range(n_classes))
    with torch.no_grad():
        for i, inputs in enumerate(test_loader):
            curr_size = inputs.size()[0]
            for j in range(curr_size):
                inputs[j] = transform(inputs[j])
            outputs = net(inputs)
            predicted = torch.argmax(outputs, dim=1)
            c = (predicted == test_labels[i*batch_size:(i+1)*batch_size]).squeeze()
            for j in range(curr_size):
                label = test_labels[i*batch_size+j]
                class_correct[int(label)] += c[j].item()
                class_total[int(label)] += 1

    for i in range(n_classes):
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i]/class_total[i]))
