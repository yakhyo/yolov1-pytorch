### YOLO v1 using PyTorch

<div align="center">
<img src="/results/res_1.jpg"/>
<img src="/results/res_2.jpg"/>
</div>

**Dataset:**
1. Download `voc2012train` [dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar)
2. Download `voc2007train` [dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar)
3. Download `voc2007test` [dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar)
4. Put all images in `JPEGImages` folder in `voc2012train` and `voc2007train` to `Images` folder as following:
```
├── Dataset 
    ├── IMAGES
        ├── 0001.jpg
        ├── 0002.jpg
    ├── LABELS
        ├── 0001.txt
        ├── 0002.txt
    ├── train.txt
    ├── test.txt
```

Each label consists of class and bounding box information. e.g `0001.txt` : 
```
1 255 247 425 468
0 470 105 680 468
1 152 356 658 754
```
**How to convert `.xml` files to `.txt` format?**
* Download [this repo](https://github.com/yakhyo/YOLO2VOC) and modify `config.py` to convert `VOC` format to `YOLO` format labels

Implementation of [YOLOv1](https://arxiv.org/pdf/1506.02640.pdf) using PyTorch

**Train:**

**Note**: I trained the backbone on IMAGENET, around ~ 10 epochs, not sure how many it was but less then 20 

```
python main.py --base_dir ../../Datasets/VOC/ --log_dir ./weights 
```

```
usage: main.py [-h] --base_dir BASE_DIR --log_dir LOG_DIR [--init_lr INIT_LR] [--base_lr BASE_LR] [--momentum MOMENTUM] [--weight_decay WEIGHT_DECAY] [--num_epochs NUM_EPOCHS]
               [--batch_size BATCH_SIZE] [--seed SEED]
```

**Evaluation:**

- `python eval.py`

- In `evaluation.py`, `im_show=False` change to `True` to see the results.

```
Evaluate the detection result...
aeroplane                 0.57
bicycle                   0.46
bird                      0.38
boat                      0.25
bottle                    0.14
bus                       0.53
car                       0.48
cat                       0.61
chair                     0.18
cow                       0.34
diningtable               0.44
dog                       0.52
horse                     0.52
motorbike                 0.49
person                    0.49
pottedplant               0.21
sheep                     0.43
sofa                      0.38
train                     0.69
tvmonitor                 0.40
mAP 0.426056536787907
```

**Detection**
- To detect objects on an image run the `detect.py` 
