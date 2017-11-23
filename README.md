# obj_tracking

## Getting Started


### prerequisites

Download data into the dataset folder. The folder structure should look like ./dataset/vot2013/xxxx
```
http://data.votchallenge.net/vot2013/vot2013.zip
```

Download model in the ./model/ folder.
```
imagenet-vgg-m.mat
```

Go to **training** folder. Run the commands below.


```
python preprocess_data.py
```




```
python train_rl.py
```
