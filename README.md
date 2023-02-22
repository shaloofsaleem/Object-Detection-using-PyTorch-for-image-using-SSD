## Object Detection 
<br>
<p align='center'>
<img src="https://github.com/Jauharmuhammed/README-Template/blob/main/assets/Mockup-website.png" width='70%' >
</p>
<br>

Object Detection using PyTorch for image using SSD | Single Shot Detection in Google Colab - Python

Object Detection in PyTorch for images. For PyTorch Object Detection, we will be using the SSD (Single Shot Detection )algorithm


<br>

### Built With

![Python](https://img.shields.io/badge/Python%20-%2314354C.svg?style=for-the-badge&logo=python&logoColor=white)



<br>

## Importing the libraries and Dataset

This is a sample for Object Detection using PyTorch for image using SSD.
PyTorch framework has excelled in tasks like object detection in Computer Vision and carrying on its legacy, we have one of the most famous Object Detection algorithms SSD. The video basically focuses on detecting objects in an image and this can easily be scaled to multiple images and even videos. We use the pretrained version of SSD meaning that there is no fine tuning involved. We also use OpenCV for making bounding boxes for the detections. The whole project is implemented in Google Colab without any GPU.
```
import torch
import torchvision
from torchvision import transforms as T
from PIL import Image
import cv2 
from google.colab.patches import cv2_imshow
```

### Prepare your dataset
 A  dataset of images with corresponding annotations that label the location and class of objects in each image. There are many datasets available online for object detection, such as COCO, PASCAL VOC, and Open Images.
 In this Project I use COCO 
 


```
COCO Dataset names:
coco_names = ["person" , "bicycle" , "car" , "motorcycle" , "airplane" , "bus" , "train" , "truck" , "boat" , "traffic light" , "fire hydrant" , "street sign" , "stop sign" , "parking meter" , "bench" , "bird" , "cat" , "dog" , "horse" , "sheep" , "cow" , "elephant" , "bear" , "zebra" , "giraffe" , "hat" , "backpack" , "umbrella" , "shoe" , "eye glasses" , "handbag" , "tie" , "suitcase" , 
"frisbee" , "skis" , "snowboard" , "sports ball" , "kite" , "baseball bat" , 
"baseball glove" , "skateboard" , "surfboard" , "tennis racket" , "bottle" , 
"plate" , "wine glass" , "cup" , "fork" , "knife" , "spoon" , "bowl" , 
"banana" , "apple" , "sandwich" , "orange" , "broccoli" , "carrot" , "hot dog" ,
"pizza" , "donut" , "cake" , "chair" , "couch" , "potted plant" , "bed" ,
"mirror" , "dining table" , "window" , "desk" , "toilet" , "door" , "tv" ,
"laptop" , "mouse" , "remote" , "keyboard" , "cell phone" , "microwave" ,
"oven" , "toaster" , "sink" , "refrigerator" , "blender" , "book" ,
"clock" , "vase" , "scissors" , "teddy bear" , "hair drier" , "toothbrush" , "hair brush"]

```
 ### SSD model
 : PyTorch provides an implementation of the SSD algorithm in the torchvision library. You can load the pre-trained SSD model from the library, or you can define and train your own SSD model from scratch using PyTorch.

```
model = torchvision.models.detection.ssd300_vgg16(pretrained = True) forms to their base form.
```
 ### data loaders:
 You'll need to define data loaders that load your dataset into memory for training and evaluation. The data loaders should handle data augmentation, such as random cropping, flipping, and scaling, to increase the diversity of your training data.
 Image link:

```
!wget http://images.cocodataset.org/val2017/000000037777.jpg
```

### Evaluate the SSD model
: After training the model, you'll need to evaluate its performance on a test set of images. You can use evaluation metrics such as mean average precision (mAP) to measure the accuracy of the model.

```
igg = cv2.imread("/content/000000037777.jpg")
for i in  range(num):
  x1, y1, x2, y2 = bbox[i].numpy().astype("int")
  igg = cv2.rectangle(igg, (x1, y1), (x2,y2),(0, 225, 0),1)
  class_name = coco_names[labels.numpy()[i] -1]
  igg = cv2.putText(igg, class_name, (x1 , y1 -10), font, 0.5, (225, 0 ,0), 1, cv2.LINE_AA)
```

### Make predictions:
Finally, you can use the trained SSD model to make predictions on new images by feeding them through the model and extracting the output bounding boxes and class labels.


<br>
## Screenshots



<table width="100%"> 
<tr>

<td width="80%">
<p align="center">
Light Mode
</p>
<img src="https://datawow.s3.amazonaws.com/blog/76/image_1/Screen_Shot_2564-02-18_at_11.20.29.png">  
</td>
</table>
<br/>


</div>
