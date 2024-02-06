# Source code for "A Purified Stacking Ensemble Framework for Cytology Classification"

[Paper link](https://link.springer.com/chapter/10.1007/978-3-031-53308-2_20)

All the experiments are conducted on GeForce RTX 3080 with TensorFlow deep learning framework. 

## 1. Dataset preparation

Arrange the dataset as follows:

```
.../dataset_name/
├── /class 1/
│   ├── xxx.png
│   ├── xxx.png
│   ├── ...
├── /class 2/
│   ├── xxx.png
│   ├── xxx.png
│   ├── ...
├── /.../
├── /class N/
│   ├── xxx.png
│   ├── xxx.png
│   ├── ...
```

Example:

```
/SIPaKMeD/
├── /im_Dyskeratotic/
│   ├── ...
├── /im_Koilocytotic/
│   ├── ...
├── /im_Metaplastic/
│   ├── ...
├── /im_Parabasal/
│   ├── ...
├── /im_Superficial-Intermediate/
│   ├── ...
```

**PS: Remember to change the keys in `utils.py/generate_csv/class2index` to your custom class names.**

## 2. Run

```bash
sh start.sh
```

**PS: You can customize shell script based on the prompts in `main.py`.**

