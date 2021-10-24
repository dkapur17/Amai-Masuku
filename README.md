# Amai-Masuku

Megathon 2021 Submission

It detects human faces with mask and no-mask even in real time. It keeps track of all people wearning and not wearing a mask in every frame.

[Weights Link](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/sidharth_giri_students_iiit_ac_in/EdcPoaXm0ZRImds55ekNgNoB2_MwChngyUsNQCN-D51eoA?e=fFzIf2)

## Running on video

Takes input of video, detects people with and without masks, and tracks them throughout the video using the centroid tracking algorithm.

Running the model:

`python3 ...`

**Output Video:** The model adds a rectangle tracker around faces and colors it based on:

- **Green:** person is wearing a mask.
- **Red:** person is not wearing a mask.

## Model

**Deep Learning Model:** We are using YOLOv4 on my own dataset. YOLOv4 achieved **93.95% mAP on Test Set**. The test set contained realistic blur images, small + medium + large faces which represent the real world images of average quality.

**YOLOv4 Training details**

- Data File = obj.data (contains training, testing and validation data)
- Cfg file = [mask.cfg](https://github.com/dkapur17/Amai-Masuku/blob/master/mask.cfg)
- Pretrained Weights for initialization= [Weights Link](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/sidharth_giri_students_iiit_ac_in/EdcPoaXm0ZRImds55ekNgNoB2_MwChngyUsNQCN-D51eoA?e=fFzIf2)
- Main Configs from yolov4-obj.cfg:
  - learning_rate=0.001
  - batch=64
  - subdivisions=64
  - steps=4800,5400
  - max_batches = 6000
  - i.e approx epochs = (6000\*64)/700 = 548
- **YOLOv4 Training results: _1.19 avg loss_**
- **Weights** of YOLOv4 trained on Face-mask Dataset: [yolov4_face_mask.weights](https://bit.ly/yolov4_mask_weights)

## Dataset

- Images were collected from [Google Images](https://www.google.com/imghp?hl=en), [Bing Images](https://www.bing.com/images/trending?form=Z9LH) and some [Kaggle Datasets](https://www.kaggle.com/vtech6/medical-masks-dataset).

- Images were annoted using [Labelimg Tool](https://github.com/tzutalin/labelImg).

- Dataset is split into 3 sets:
  | _Set_ | Number of images | Objects with mask | Objects without mask |
  | :----------------: | :--------------: | :---------------: | :------------------: |
  | **Training Set** | 700 | 3047 | 868 |
  | **Validation Set** | 100 | 278 | 49 |
  | **Test Set** | 120 | 503 | 156 |
  | **Total** | 920 | 3828 | 1073 |

- **Download the Dataset here**:

  - [Github Link](https://github.com/adityap27/face-mask-detector/tree/master/dataset) or
  - [Kaggle Link](https://www.kaggle.com/aditya276/face-mask-dataset-yolo-format)

## Performance

- Below is the comparison of YOLOv4 on 3 sets.
- **Metric is mAP@0.5** i.e Mean Average Precision.
- **Frames per Second (FPS)** was measured on **Google Colab GPU - Tesla P100-PCIE** using **Darknet** command: [link](https://github.com/AlexeyAB/darknet#how-to-evaluate-fps-of-yolov4-on-gpu)

|                                                     Model                                                     | Training Set | Validation Set | Test Set |
| :-----------------------------------------------------------------------------------------------------------: | :----------: | :------------: | :------: |
| [YOLOv4](https://github.com/adityap27/face-mask-detector/blob/master/media/YOLOv4%20Performance.jpg?raw=true) |    99.65%    |     88.38%     |  93.95%  |

- Yolov4 achieves good performance as it has **Low bias** and **Medium Variance**.

## Can we do better?

- As described earlier that YOLOv4 is giving 93.95% mAP on Test Set, this can be improved in the following ways:

  1. Use more Training Data.
  2. Use more Data Augmentation for Training Data.
  3. Train with larger network-resolution by setting your `.cfg-file` (height=640 and width=640) (any value multiple of 32).
  4. For Detection use even larger network-resolution like 864x864.
  5. Try YOLOv5 or any other Object Detection Algorithms like SSD, Faster-RCNN, RetinaNet, etc. as they are very good as of now (year 2020).

## References

- [YOLOv4 Paper](https://arxiv.org/abs/2004.10934)
- [Darknet github Repo](https://github.com/AlexeyAB/darknet)
- [YOLO Inference with GPU](https://www.pyimagesearch.com/2020/02/10/opencv-dnn-with-nvidia-gpus-1549-faster-yolo-ssd-and-mask-r-cnn/)
