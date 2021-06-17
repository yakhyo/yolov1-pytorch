### YOLO v1 using PyTorch


**Dataset:**
1. Download `voc2012train` [dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar)
2. Download `voc2007train` [dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar)
3. Download `voc2007test` [dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar)
4. Put all images in `JPEGImages` folder in `voc2012train` and `voc2007train` to `Images` folder as following:
```
├── Dataset 
    ├── Images
        ├── 0001.jpg
        ├── 0002.jpg
    ├── Labels
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


**Train:**
- `python main.py`

**Evaluation:**
- `python eval.py`
- In `evaluation.py`, `im_show=False` change to `True` to see the results.
