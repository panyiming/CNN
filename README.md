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

|operation|inshape|outshape|params|time|
|:--:|:--:|:--:|:--:|:--:|
||||||
