---
title: Anomaly detection on the MVTec AD database
feed: show
date: 2026-08-21
---

In this project, I tried to perform **zero-shot** anomaly detection on the MVTec database. By **zero-shot** we meant that we only train with model with nominal/anomaly-free pictures of the subjects but then we want to be able to distinguish between nominal and anomaly. The MVTec dabase consists on series of pictures of

- bottle
- cable
- capsule
- carpet
- grid
- hazelnut
- leather
- metal_nut
- pill
- screw
- tile
- toothbrush
- wood
- zipper

which of these datasets contains a `train` folder which consists only of nominal pictures; and a `test` folder that consists of a `good` folder (nominal pictures) and other folders labeled with specific anomalies. E.g. for `bottle`, the anomaly folders are `broken_large` , `broken_small` and  `contamination`. We do not plain to distinguish between each type of anomaly, only whether the image contain an anomaly or not.

Due to computational and time limitations, I will only focus on three categories

- bottle - circular symmetry, all `train`images are very similar

![[0d53fefd58e59e2f2b6a3e9fe510c2d82e168bb7.png|400]]

- carpet - translational invariance and same pattern with slightly different thread colours

![[a08db728849ec2e11b919d4297a17a7378a0e1d9.png|400]]

- hazelnut - similar object rotated around, i.e. orientation is random

![[bcb9ab83b421c488a71c01a5565de680a107f178.png|400]]
# Methods

I will try to find the anomalies using two, quite different, algorithsm, which are **Convolutional Auto encoders (CAE)** and **Patchcore**. As we will soon see, CAE perform particularly bad in this dataset and patchcore shows amazing results. If needs be, we will select the threshold that maximizes the `f1_score`.

## **Convolutional Auto encoders (CAE)**

The idea is to have a convolutional network that takes the input image, compresses it down to a latent space and then tries to reconstruct it back to the original image. The latent space has a lower dimension compared to the input image so the network has to learn the patterns of the object so that the reconstructed image is as close to the original, that is the core idea of auto encoders. Since we only use nominal data to train the CAE, the idea is for the CAE to learn how to reconstruct nominal data but not anomalous data, so we will see an increase on the loss for anomalous data compared to nominal.

We choose between $6$ different CNN architectures, some of which were acquired from articles, suggested by LLMs or some I came up by trying to improve performance.

A disclaimer is necessary. Training was not done properly, I should have done k-fold to remove sampling bias; and trained for how many epochs were necessary to reach a minimum of the loss function. Moreover, each network could have been optimised for each dataset, such was not done. None of this was done due to time/computational constraints. The loss function selected was `MSE` and we used an Adam optimiser.

I trained one CAE for many epochs and results improved significantly, unfortunately the same could not be done for all datasets/networks as it would take an unfeasible amount of time so. For the sake of comparison, we only train each of these networks for $100$ epochs.

### CAE architectures

```python
# Mine 1
class Autoencoder_000(nn.Module):
    def __init__(self):
        super(Autoencoder_000,self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5), nn.BatchNorm2d(6), nn.ELU(True),
            nn.Conv2d(6, 6, kernel_size=5), nn.BatchNorm2d(6), nn.ELU(True),
            nn.Conv2d(6, 6, kernel_size=5), nn.BatchNorm2d(6), nn.ELU(True),
            nn.Conv2d(6, 2, kernel_size=5), nn.BatchNorm2d(2), nn.ELU(True))
        self.decoder = nn.Sequential(             
            nn.ConvTranspose2d(2, 6, kernel_size=5), nn.BatchNorm2d(6), nn.ELU(True),
            nn.ConvTranspose2d(6, 6, kernel_size=5), nn.BatchNorm2d(6), nn.ELU(True),
            nn.ConvTranspose2d(6, 6, kernel_size=5), nn.BatchNorm2d(6), nn.ELU(True),
            nn.ConvTranspose2d(6, 3, kernel_size=5),
            torch.nn.Sigmoid())
    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
```

```python
# Claude model 2
class Autoencoder_001(nn.Module):
    def __init__(self):
        super(Autoencoder_001,self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3,  32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ELU(True),  # 32→16
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ELU(True),  # 16→8
            nn.Conv2d(64, 2,  3, stride=1, padding=1))                                     # no BN/act
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2,  64, 3, stride=1, padding=1), nn.BatchNorm2d(64), nn.ELU(True),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(32), nn.ELU(True),
            nn.ConvTranspose2d(32, 3,  3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid())
        
    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
```

```python
# Mine 2
class Autoencoder_002(nn.Module):
    def __init__(self):
        super(Autoencoder_002,self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 6, stride=2, padding=1, kernel_size=5), nn.BatchNorm2d(6), nn.ELU(inplace=True),
            nn.Conv2d(6, 6, stride=2, padding=1, kernel_size=5), nn.BatchNorm2d(6), nn.ELU(inplace=True),
            nn.Conv2d(6, 6, stride=2, padding=1, kernel_size=5), nn.BatchNorm2d(6), nn.ELU(inplace=True),
            nn.Conv2d(6, 2, stride=1, padding=1, kernel_size=5), nn.BatchNorm2d(2), nn.ELU(inplace=True))
        self.decoder = nn.Sequential(             
            nn.ConvTranspose2d(2, 6, padding=1, stride=1, kernel_size=5), nn.BatchNorm2d(6), nn.ELU(inplace=True),
            nn.ConvTranspose2d(6, 6, padding=1, stride=2, kernel_size=5), nn.BatchNorm2d(6), nn.ELU(inplace=True),
            nn.ConvTranspose2d(6, 6, padding=1, stride=2, kernel_size=5), nn.BatchNorm2d(6), nn.ELU(inplace=True),
            nn.ConvTranspose2d(6, 3, padding=1, stride=2, kernel_size=5, output_padding=1),
            torch.nn.Sigmoid())
    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
```

```python
# Claude suggestion with modifications
class Autoencoder_003(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3,  32, 4, stride=2, padding=1), nn.BatchNorm2d(32), nn.ELU(), 
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.BatchNorm2d(64), nn.ELU(),  
            nn.Conv2d(64, 2,  3, stride=1, padding=1))                               
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2,  64, 3, stride=1, padding=1), nn.BatchNorm2d(64), nn.ELU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.BatchNorm2d(32), nn.ELU(),  
            nn.ConvTranspose2d(32, 3,  4, stride=2, padding=1),                                
            nn.Sigmoid())

    def forward(self, x):
        return self.decoder(self.encoder(x))
```

```python
# Claude suggestion with modifications
class Autoencoder_004(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3,  32, 4, stride=2, padding=1), nn.BatchNorm2d(32), nn.ELU(), 
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.BatchNorm2d(64), nn.ELU(),  
            nn.Conv2d(64, 2,  3, stride=3, padding=1), nn.BatchNorm2d(2), nn.ELU())
        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(2,  64, 3, stride=3, padding=1, output_padding=1), nn.BatchNorm2d(64), nn.ELU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.BatchNorm2d(32), nn.ELU(),  
            nn.ConvTranspose2d(32, 3,  4, stride=2, padding=1),                                
            nn.Sigmoid())

    def forward(self, x):
        return self.decoder(self.encoder(x))
```

```python
# https://www.geeksforgeeks.org/machine-learning/implement-convolutional-autoencoder-in-pytorch-with-cuda/
class Autoencoder_005(nn.Module):
    def __init__(self):
        super(Autoencoder_005, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(16, 8, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2,
                               padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2,
                               padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
```

## Patchcore (https://arxiv.org/abs/2106.08265)

**Patchcore** - The idea between patchcore is to use a pre trained netwoork, e.g. `resnet50`, to generates new features about the input image using the intermediate representation of the image in its layers. The argument is that these intermediate layers that capture important abstract patterns/characteristics that better generalize for any object. With these features we construct a memory bank of the feature map of only the nominal data. When doing anomaly detection, we measure if that image feature bank is close to what we have in the memory bank and, if it is not, we signal it an anomaly. This algorithm also produces a segmentation map, i.e. it can locate where the anomaly is located.

Disclaimer: From their paper, we already know that patchcore perfects exceptionally well.

## MLP auto encoders

My first attempt was to use MLP auto encoders but, for the `bottle` dataset where most of the images are very similar, the network ended up memorising the output image. I even tried random latent space configurations but all yielded the same results. This made prediction terrible as it was, basically, a difference between the target image and *some for of mean* of the training images. For this reason, this architecture was not used, although it could in principle be used for the `carpet` and for the `hazelnut`.

# Bottle

## CAE

The loss function of each model clearly shows that we should have trained for longer, as loss is still decreasing.

![[147c7ca1aad3a942b5e01addbfcb03f9cbb13f9d.png|500]]

By analysing the AUC of the ROC, we see that we `Autoencoder_002` reaches `AUC = 0.8`.

    model                 AUC
    ---------------  --------
    Autoencoder_000  0.721429
    Autoencoder_001  0.599206
    Autoencoder_002  0.802381
    Autoencoder_003  0.61746
    Autoencoder_004  0.719841
    Autoencoder_005  0.562698

It is curious to note that `Autoencoder_002` is not the model with the lowest loss, which signals that the loss function but not be a good indicator for a good anomaly detector. The reason is that it does not really matter if the model can reconstruct the image, it only matter if it was a difference performance reconstructing anomalous data compared to nominal data.

![[d135b52da3edfe3b28ea2d06c68958acb6eb5553.png|500]]

I will also show the image, the reconstructed image and the difference for an anomalous picture.

![[03f8a7a4648426093754c76e0c7f459584648415.png]]

Where we see that the reconstructed image is blurry which misses fine details of the defect. Other models reproduce crispy images, and even manage to recreate the defects. This is an indication that the latent space/bottleneck is not small enough, and that maybe should also try a more shallow network.

## Patchcore

It is already known that patchcore performs very well in this dataset, building it is also somewhat quick as no training is necessary. We will instead do a hyperparameter search so that we have low inference times and high precision. We will vary the

- `backbone` - `resnet50`, `wide_resnet50_2` - where the feature map come from
- `coreset_sampling_ratio` - to improve inference speed, only a fraction .`coreset_sampling_ratio`of the memory bank is taken into account.
- `num_neighbors` - `[2, 6, 10]` - during inference, we use a KNN to rescale the anomaly score with `num_neighbors` many neighbours.

For the `bottle`, the results are

    id    separation       roc    neightbors    coreset_ratio  backbone
    ----  ------------  --------  ------------  ---------------  ----------
       0     -1         0.673016             2            0.001  resnet50
       1      0         0.5                  6            0.001  resnet50
       2      0         0.5                 10            0.001  resnet50
       3     -0.297607  0.928571             2            0.001  wide
       4      0.487549  1                    6            0.001  wide
       5      0.499756  1                   10            0.001  wide
       6     -0.323975  0.993651             2            0.01   resnet50
       7      0.26123   1                    6            0.01   resnet50
       8      0.273926  1                   10            0.01   resnet50
       9     -0.25293   0.975397             2            0.01   wide
      10      0.27124   1                    6            0.01   wide
      11      0.295166  1                   10            0.01   wide
      12     -0.65332   0.902381             2            0.1    resnet50
      13      0.223145  1                    6            0.1    resnet50
      14      0.251465  1                   10            0.1    resnet50
      15     -0.13147   0.988889             2            0.1    wide
      16      0.297119  1                    6            0.1    wide
      17      0.285889  1                   10            0.1    wide

where we see that we have many models with `AUC = 1`which means that they are perfect anomaly detectors, of course this might change with more data. Overall, we have that the `wide_resnet50_2` performs better than the `resnet50`; and that `num_neighbors` should be at least $4$. We see terrible performance for `coreset_ratio = 0.001` and `resnet50`. In order to distinguish between all the models with `AUC = 1`, we will introduce a new metric `seperation` that calculates the distance of the anomaly score between the nominal picture with the highest score and the anomalous picture with the lowest score. This metric will tell us how good the model is at separating the nominal from the anomalous, so the bigger this metric is the better. A negative separation only occurs for non-perfect anomaly detectors.
We then find that the best model is

    num_neighbors = 2, coreset_ratio = 0.001, backbone = `wide_resnet50_2`

which interestingly, has a smaller memory bank compared to the other models. It appears that more memory items introduce variance that make the model less certain.

# Carpet

## CAE

The loss function of each model clearly shows that we should have trained for longer, as loss is still decreasing.

![[477b1b99e8d67ee8a82ce289fa4c3edaf7255729.png|500]]

I will show now, the `AUC` for each model

    model                 AUC
    ---------------  --------
    Autoencoder_000  0.32183
    Autoencoder_001  0.371589
    Autoencoder_002  0.418138
    Autoencoder_003  0.361958
    Autoencoder_004  0.408909
    Autoencoder_005  0.364366

and we can see that all of them are less than $0.5$ (random chance), and that more epochs are different architectures are needed.

Showing the `ROC` curve of the *best* performing model, `Autoencoder_003`, shows that for most threshold values, it would be better to go against the model (or flip a coin). For some threshold values, this model is very slightly better than flipping a coin.

![[97411d62c7abedf544ac21bf7adb01d1ff673f7a.png|500]]

I will also show the image, the reconstructed image and the difference for an anomalous picture.

![[e4f93065b37ce3022cef7bc270e576abdc2d9928.png]]

Where the anomaly is reconstructed, so it is necessary to increase tighten the bottleneck on these models. For this image in particular, all models reconstructed the anomaly, except model `Autoencoder_004` that just reconstructs the carpet without anomaly but contains a lot of noise that spoils the `MSE`, which makes the anomaly go on undetected.

## Patchcore

It is already known that patchcore performs very well in this dataset, building it is also somewhat quick as no training is necessary. We will instead do a hyperparameter search so that we have low inference times and high precision. We will vary the

- `backbone` - `resnet50`, `wide_resnet50_2` - where the feature map come from
- `coreset_sampling_ratio` - to improve inference speed, only a fraction .`coreset_sampling_ratio`of the memory bank is taken into account.
- `num_neighbors` - `[2, 6, 10]` - during inference, we use a KNN to rescale the anomaly score with `num_neighbors` many neighbours.

For the `carpet`, the results are

    id    separation       roc    neightbors    coreset_ratio  backbone
    ----  ------------  --------  ------------  ---------------  ----------
       0     -1         0.53451              2            0.001  resnet50
       1      0         0.5                  6            0.001  resnet50
       2      0         0.5                 10            0.001  resnet50
       3     -1         0.788925             2            0.001  wide
       4     -0.231934  0.64988              6            0.001  wide
       5     -0.160156  0.650682            10            0.001  wide
       6     -1         0.623796             2            0.01   resnet50
       7      0         0.5                  6            0.01   resnet50
       8      0         0.5                 10            0.01   resnet50
       9     -1         0.868178             2            0.01   wide
      10     -0.623047  0.856942             6            0.01   wide
      11     -0.54126   0.856942            10            0.01   wide
      12     -0.88501   0.838082             2            0.1    resnet50
      13     -0.244629  0.57805              6            0.1    resnet50
      14     -0.151367  0.542335            10            0.1    resnet50
      15     -1         0.917135             2            0.1    wide
      16     -0.697266  0.94382              6            0.1    wide
      17     -0.695312  0.926164            10            0.1    wide

where we can see that we do not have a perfect model with `AUC = 1`. In general, the `wide_resnet50_2` performs better than `resnet50`, and `num_neighbors` seems to be prefered at either $2$ or $6$.

The goal is to find the model and a threshold that maximizes the `f1_score`. That model is
`num_neighbors = 6, coreset_sampling_ratio = 0.1, backbone = wide_resnet50_2`

with a threshold of `1.0` (i.e. exactly $1$, one should subtract a small number to account for numerical fluctiations) with an `f1_score` of $0.97$.

In the cases where inference time is of great importance, we can trade this model for
`num_neighbors = 2, coreset_sampling_ratio = 0.01, backbone = wide_resnet50_2`
which has a lower `AUC` ($0.94 \to 0.87$) but it many times faster at inference. For this model, the optimal threshold of still `1.0` with an `f1_score` of $0.92$.

# Hazelnut

## CAE

The loss function of each model clearly shows that we should have trained for longer, as loss is still decreasing.

![[d6231d4e0e9396a9a39110e652c088ae39b0a06c.png|500]]

I will show now, the `AUC` for each model

    model                 AUC
    ---------------  --------
    Autoencoder_000  0.691071
    Autoencoder_001  0.769286
    Autoencoder_002  0.7925
    Autoencoder_003  0.741071
    Autoencoder_004  0.879286
    Autoencoder_005  0.857857

where we get decent `AUC`. The best model `Autoencoder_004` reaches `AUC = 0.88`.

Showing the `ROC` curve of the *best* performing model, `Autoencoder_004`, shows that the model performs quite well.

![[f1019bdcb6299a6ac5a99a14f3fb5d5b6d901c33.png|500]]

I will also show the image, the reconstructed image and the difference for an anomalous picture.

![[84a8551ef22cbd793ccde15694f96f59e8efefb7.png]]

Where the anomaly is still reconstructed but the whole image appears more blurry.

## Patchcore

It is already known that patchcore performs very well in this dataset, building it is also somewhat quick as no training is necessary. We will instead do a hyperparameter search so that we have low inference times and high precision. We will vary the

- `backbone` - `resnet50`, `wide_resnet50_2` - where the feature map come from
- `coreset_sampling_ratio` - to improve inference speed, only a fraction .`coreset_sampling_ratio`of the memory bank is taken into account.
- `num_neighbors` - `[2, 6, 10]` - during inference, we use a KNN to rescale the anomaly score with `num_neighbors` many neighbours.

For the `hazelnut`, the results are

    id    separation       roc    neightbors    coreset_ratio  backbone
    ----  ------------  --------  ------------  ---------------  ----------
       0   -0.821777    0.978929             2            0.001  resnet50
       1    0.0419922   1                    6            0.001  resnet50
       2    0.0419922   1                   10            0.001  resnet50
       3   -0.267334    0.980714             2            0.001  wide
       4    0.185303    1                    6            0.001  wide
       5    0.185303    1                   10            0.001  wide
       6   -0.767578    0.960714             2            0.01   resnet50
       7    0.0810547   1                    6            0.01   resnet50
       8    0.0810547   1                   10            0.01   resnet50
       9   -0.617676    0.953571             2            0.01   wide
      10    0.29834     1                    6            0.01   wide
      11    0.30127     1                   10            0.01   wide
      12   -0.42749     0.849464             2            0.1    resnet50
      13    0.00341797  1                    6            0.1    resnet50
      14    0.0356445   1                   10            0.1    resnet50
      15   -0.340576    0.925893             2            0.1    wide
      16    0.271729    1                    6            0.1    wide
      17    0.278809    1                   10            0.1    wide

where we have many perfect models, except when `num_neighbors` = 2. Looking at separation, i.e. lowest anomalous anomaly score minus highest nominal anomaly score, we see a difference between the models. The model with the highest separation is
`num_neighbors = 10, coreset_sampling_ratio = 0.01, backbone = wide_resnet50_2`
with a seperation of $0.3$.

Again, if inference speed is important, one could trade this model for
`num_neighbors = 6/10, coreset_sampling_ratio = 0.001, backbone = wide_resnet50_2`
which has a separation on $0.18$, i.e. comparable with $0.3$, but has an inference speed many times faster than the previous model.
