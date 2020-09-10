# FoodRecognition

1. Download the dataset at this link: https://www.kaggle.com/kmader/food41 (5GB).
2. Change the path on the dataset in _food_recognition.py_, _tools.py_ and _nnet.py_.
2. Run _food_recognition.py_.

## If you want to change some settings... 
- In _tools.py_ under _loadImgaes()_ you can change the size of the images. If you make this change, you also need to update the input value of the first linear layer in _nnet.py_.
- In _food_recognition.py_ you can change the variable _n_images_for_class_ to modify the number of images selected for train and test set. 
- At the beginning of _knn.py_, _svm.py_ and _nnet.py_ you can change some inital parameters to see different kind of performances. 
