# CNN implementation with Numpy

### 1 Install

Firstly, please make sure your python version is python3.

#### 1.1 clone code

```shell
git clone https://github.com/panyiming/cnn.git
```

#### 1.2 pip install packages

```shell
pip install -r requirements.txt -i  https://pypi.tuna.tsinghua.edu.cn/simple/
```

### 2 Get data

Open 'data' folder

#### 2.1 unzip files

```
gzip -d *gz
```

#### 2.2 generate data

```
python load.py
```

### 3 Train and test

#### 3.1 train

```shell
sh run.sh
```

#### 3.2 test

```shell
sh run_test.sh
```
