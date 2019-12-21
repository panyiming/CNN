# CNN implementation with Numpy

### 1 Install

Firstly, please make sure your python version is python3.

#### 1.1 Clone code

```shell
git clone https://github.com/panyiming/cnn.git
```

#### 1.2 Pip install packages

```shell
pip install -r requirements.txt -i  https://pypi.tuna.tsinghua.edu.cn/simple/
```

### 2 Get data

Run following command in data folder.

#### 2.1 Unzip files

```shell
gzip -d *gz
```

#### 2.2 Generate data

```shell
python load.py
```

### 3 Train and test

#### 3.1 Train

```shell
sh run.sh
```

#### 3.2 Test

```shell
sh run_test.sh
```

### 4 The effect of im2col

|operation|im2col|inshape|outshape|params|time(ms)|
|:--:|:--:|:--:|:--:|:--:|:--:|
|convolution forward|no|(128, 3, 112, 112)|(128, 16, 110, 110)|kenerl=(3, 3), stride=1, pad=0|12.3799|
|convolution forward|yes|(128, 3, 112, 112)|(128, 16, 110, 110)|kenerl=(3, 3), stride=1, pad=0|0.5719|
|convolution backward|no|(128, 16, 110, 110)|(128, 3, 112, 112)|kenerl=(3, 3), stride=1, pad=0|31.4386|
|convolution backward|yes|(128, 16, 110, 110)|(128, 3, 112, 112)|kenerl=(3, 3), stride=1, pad=0|4.6786|
|maxpool forward|no|(128, 16, 112, 112)|(128, 16, 56, 56)|kenerl=(2, 2), stride=2, pad=0|1.7381|
|maxpool forward|yes|(128, 16, 112, 112)|(128, 16, 56, 56)|kenerl=(2, 2), stride=2, pad=0|2.9103|
|maxpool backward|no|(128, 16, 56, 56)|(128, 16, 112, 112)|kenerl=(2, 2), stride=2, pad=0|0.7157|
|maxpool backward|yes|(128, 16, 56, 56)|(128, 16, 112, 112)|kenerl=(2, 2), stride=2, pad=0|6.0505|
