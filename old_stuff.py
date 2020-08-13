n_images_per_class = 75 # len(list(train_set.keys())[0]) 750
n_images = len(train_set) * n_images_per_class 
name = 'train_features_' + str(n_images_per_class) + '.csv'    
if os.path.isfile(name):
    train_labels = extractLabels(train_set, n_images, n_images_per_class)
    train_feat = np.genfromtxt(name, delimiter=',')
else:
    train_labels, train_feat = extractFeatures(train_set, n_images, n_images_per_class)
    with open(name, 'w', newline='') as filename:
        writer = csv.writer(filename)
        writer.writerows(train_feat)
    
n_images_per_class = 25 # len(list(train_set.keys())[0]) 250
n_images = len(test_set) * n_images_per_class  
name = 'test_features_' + str(n_images_per_class) + '.csv' 
if os.path.isfile(name):
    test_labels = extractLabels(test_set, n_images, n_images_per_class)
    test_feat = np.genfromtxt(name, delimiter=',')

else:
    test_labels, test_feat = extractFeatures(test_set, n_images, n_images_per_class)
    with open(name, 'w', newline='') as filename:
        writer = csv.writer(filename)
        writer.writerows(test_feat)


def extractFeatures(t_set, n_images, n_images_per_class):

    # Extractor initialization
    extractor = FeaturesExtractor()

    # Infer from all images (or until one is reached)
    n_features = 2048
    labels = []
    features = np.zeros((n_images, n_features))
    idx = -1
    for c in t_set:
        print(c)
        img_per_set = 0
        for name in t_set[c]:
            print(img_per_set, end=' ')
            idx += 1
            img_per_set += 1
            labels.append(name)
            img = cv2.imread(os.path.join(path_img, name))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img,(128,128))
            # img_list size has to be (N,W,H,C), output has size (N,2048)
            features[idx] = extractor.getFeatures(img)
            if img_per_set == n_images_per_class:
                break
        print()
        if c == 'bruschetta': # ONLY THE FIRST 11 CLASSES
            break

    #print(features[:5,:])
            
    return np.array(labels), features


def extractLabels(t_set, n_images, n_images_per_class):
    labels = []
    for c in t_set:
        img_per_set = 0
        for name in t_set[c]:
            img_per_set += 1
            labels.append(c)
            if img_per_set == n_images_per_class:
                break
    return np.array(labels)