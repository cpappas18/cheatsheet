This document contains refreshers and short explanations on the concepts. For a more in-depth explanation, please see the sources at the end of the document.

# Table of contents
1. Regression models
2. Classification models
3. Computer vision  
  a. Semantic image segmentation models

# Computer vision
## Semantic image segmentation models

1. **Pixel accuracy** is the percentage of pixels in your image that are classified correctly. 
The issue with this metric is that if you have *class imbalance* (one or more classes dominate the image), 
you can get a good pixel accuracy score if the dominating classes are correctly classified, even if the other classes aren't.  

2. **Intersection-Over-Union (IOU) Index**, also known as the **Jaccard Index**, is the area of intersection between the prediction 
and the ground truth *divided* by the area of union between the prediction and the ground truth. The mean IoU of an image is the
average of the IoU of each class.
```
# IoU Keras implementation example - source at the end of this document  

from keras import backend as K
def iou_coef(y_true, y_pred, smooth=1):
  intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
  union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
  iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
  return iou
```

3. **Dice coefficient**, or F1 Score, is *2 x* the area of overlap divided by the total number of pixels in both images (predicted
and ground truth). The mean Dice coefficient of an image is the average of the Dice coefficients of each class.

```
# Dice implementation example - source at the end of this document  

def dice_coef(y_true, y_pred, smooth=1):
  intersection = K.sum(y_true * y_pred, axis=[1,2,3])
  union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
  dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
  return dice
```

**Sources:**  
[Semantic image segmentation performance](https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2)
