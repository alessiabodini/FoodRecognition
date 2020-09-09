import os
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision import transforms as tr

path = Path(os.path.join('C:/', 'Users', 'ale19', 'Downloads', 'Food-101')) # your path of the dataset
path_h5 = path
n_classes = 10
learning_rate = 1e-2 # 1e-3 or 1e-2
batch_size = 4
epochs = 2 # 2 or 5 or 10

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 14 * 14, 2048) # 64x64: 14*14, 128x128: 30*30
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


def nnet(train_set, test_set, train_labels, test_labels, classes):

    # 1. Shuffle train and test set
    train_set = np.transpose(train_set, (0, 3, 1, 2))
    test_set = np.transpose(test_set, (0, 3, 1, 2))
    train_set = train_set.astype('float32')
    test_set = test_set.astype('float32')
    #print('Dataset loaded.')

    indices = np.arange(train_set.shape[0])
    np.random.shuffle(indices)
    train_set = train_set[indices]
    train_labels = train_labels[indices]

    indices = np.arange(test_set.shape[0])
    np.random.shuffle(indices)
    test_set = test_set[indices]
    test_labels = test_labels[indices]

    # 2. Transform train and test set into tensors
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    train_labels = torch.tensor(train_labels, dtype=torch.long)
    test_labels = torch.tensor(test_labels, dtype=torch.long)

    # Use torch.true_divide(image, 255) insted
    #transform = tr.Normalize((128, 128, 128), (127, 127, 127)) 

    # 3. Create a model
    net = Net()

    # 4. Define a loss function and the optimizer
    criterion = nn.CrossEntropyLoss() # or MSELoss 
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    #optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # 5. Train the network (if necessary)
    net.train()
    if os.path.isfile('.pth'):
        net.load_state_dict(torch.load('.pth'))
    
    else:
        for epoch in range(epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, inputs in enumerate(train_loader):
                curr_size = inputs.size()[0]

                for j in range(curr_size):
                    inputs[j] = torch.true_divide(inputs[j], 255)
                outputs = net(inputs)
                
                #right_prob = torch.zeros([curr_size, n_classes], dtype=torch.float32)
                #for j in range(curr_size):
                #    right_prob[j, int(train_labels[i*batch_size+j])] = 1
                loss = criterion(outputs, train_labels[i*batch_size:(i+1)*batch_size])
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                # Print statistics
                running_loss += loss.item()
                if i % 100 == 0:
                    print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / 100))
                    running_loss = 0.0

        print('Training ended.')

        # Save the obtained model
        torch.save(net.state_dict(), 'trained_network.pth')


    # 6. Test the network
    net.eval() # disable batch normalization
    cmc = np.zeros((n_classes, n_classes))
    correct = 0
    total = 0
    total_predicted = []
    with torch.no_grad():
        for i, inputs in enumerate(test_loader):
            curr_size = inputs.size()[0]
            for j in range(curr_size):
                inputs[j] = torch.true_divide(inputs[j], 255)
            outputs = net(inputs)
            predicted = torch.argmax(outputs, dim=1)
            total_predicted.extend(predicted)
            total += curr_size
            correct += (predicted == test_labels[i*batch_size:(i+1)*batch_size]).sum().item()
            for predict, test_label in zip(predicted, test_labels[i*batch_size:(i+1)*batch_size]):
                cmc[int(test_label), int(predict)] += 1.0
            #if i % 10 == 0:
                #print(predicted)
                #print(test_labels[i*batch_size:(i+1)*batch_size], end='\n')

    precision = []
    recall = []
    for i in range(n_classes):
        if cmc[i,i] != 0:
            precision.append(cmc[i,i] / np.sum(cmc[:,i]))
            recall.append(cmc[i,i] / np.sum(cmc[i,:]))

    precision = np.mean(np.asarray(precision))
    recall = np.mean(np.asarray(recall))

    print('Accuracy of the network on the test images: {0:.2f} %'.format(100 * correct/total))
    print('Classifier\'s mean precision: ' + "{0:.2f}".format(precision))
    print('Classifier\'s mean recall: ' + "{0:.2f}".format(recall))


    # 7. Test performance for every class
    class_correct = list(0. for i in range(n_classes))
    class_total = list(0. for i in range(n_classes))
    with torch.no_grad():
        for i, inputs in enumerate(test_loader):
            curr_size = inputs.size()[0]
            for j in range(curr_size):
                inputs[j] = torch.true_divide(inputs[j], 255)
            outputs = net(inputs)
            predicted = torch.argmax(outputs, dim=1)
            c = (predicted == test_labels[i*batch_size:(i+1)*batch_size]).squeeze()
            for j in range(curr_size):
                label = test_labels[i*batch_size+j]
                class_correct[int(label)] += c[j].item()
                class_total[int(label)] += 1

    print()
    for i in range(n_classes):
        print('Accuracy of %5s: %2d %%' % (classes[i], 100 * class_correct[i]/class_total[i]))

    print()
    return total_predicted