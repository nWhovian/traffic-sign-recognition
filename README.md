# Traffic Sign Classification

## Introduction
One of the possible solutions to the traffic sign recognition problem is described here. The classification algorithm takes a sign image with a small context around it and classifies which of 43 sign types it belongs to. Let us describe in more detail this classification system.

## Data preprocessing
Work begins with data reading and its preprocessing. Some of the presented pictures are square, others are not, so first of all, we must bring all the images to a single shape (I use the OpenCV to perform this). For example, Figure 1 is converted to a square shape - Figure 2.
![](https://imgur.com/noK5kXr.png)

After that we need to resize the images to the same dimensionality (example: Figure 3 is obtained by resizing Figure 2 to 30x30 px)
![](https://imgur.com/1kZZZcz.png)

After processing the images themselves, we will now focus on the entire dataset. From Figure 4 we see that the distribution of images for each class is uneven. For better performance, we will perform an augmentation: randomly modify existing pictures and add them as new data to the same class so that the number of examples in all classes is equal (Figure 5)
![](https://imgur.com/Cyhu2uM.png)

In this case, the original images were subjected to several types of augmentation with random coefficients, which seemed to me the most suitable for the problem of recognizing the traffic signs. 
This is a **rotation** from -45 to 45 (because the real signs may well be rotated that way), **adding random noise** to the image (because the cameras and transmitting information are not perfect, and noise appears in real life), **rescaling the image intensity** (because it is natural that among the real data there will be pictures with very different brightness and saturation: from night images to those taken in a bright sunlight) and **blurring pictures** (as a representation of a poor quality shooting). 
Here are the examples of a) rotation, b) noise adding, c) intensity rescaling, d) blur image:
![](https://imgur.com/aj1oArn.png)
![](https://imgur.com/ZbEc2q1.png)
![](https://imgur.com/WAuj5IT.png)
![](https://imgur.com/sNzRCda.png)
These techniques can be applied not only individually but also together, for example:
![](https://imgur.com/Mr9J4et.png)

It remains only to normalize the values for all images and translate the matrix representation into a one-dimensional vector using the numpy.ravel function.

## Training & Testing
Now we train **RandomForestClassifier** and evaluate the predicted values on the validation and test sets & recall and precision:
```
validation_accuracy = 0.76 
test_accuracy = 0.735
```

## Experiments
We said earlier that on a heterogeneous dataset, the augmentation (generation of synthetic data) is necessary. But how will the result change if this augmentation is not carried out? Let us evaluate the same model without synthetic data generation step:
```
test_accuracy = 0.76
```
It turns out that augmentation increases the accuracy score (not much, but still) and is useful to apply.
Let us now try to change the size of preprocessed images and see what happens.
As a result of calculations, we get the following numbers:
```
10x10: time - 88.6205, score - 0.6136 
20x20 : time - 136.4541, score - 0.71045 
30x30: time - 289.7632, score - 0.73365 
40x40: time - 337.2045, score - 0.7476 
50x50: time - 558.6452, score -0.7515
```
![](https://imgur.com/52gcTMB.png)

Thus, we clearly see that the accuracy depends on the size of the image (which only confirms the thoughts on the mistakes of the classifier). In turn, the runtime and image size are not interdependent. Time depends on many parameters and varies greatly from iteration to iteration, in opposite to accuracy.

## Conclusion
To summarize: we wrote an algorithm that solves the traffic sign recognition problem using RandomForestClassifier and augmentation techniques and did the performance analysis. We also found out how the parameter of data preprocessing (image size) and the generation of synthetic data affect the result.
I think it’s also worth trying to add image cropping to the image preprocessing so that too much background does not fall into the frame; and also improve data transformation within the augmentation.
