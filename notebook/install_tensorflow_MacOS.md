# Setup environment and install Tensorflow on MacOS
## 1. Python3 installed?
Although Python 2 is installed by default on Apple computers, Python 3 is not. But it's worth a check before we move on. You can check by running these command on Terminal:

```
$ python --version
Python 2.7.10
```

```
$ python3 --version
python3: command not found
```

An error message should be returned for the latter. Otherwise, you have ```python3``` already installed on your computer.

## 2. Install Xcode and Homebrew

```
$ xcode-select --install
```

```
$ /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
```
Check if Homebrew installed correctly:

```
$ brew doctor
Your system is ready to brew.
```

## 3. Install Python 3

The most recent ```Python``` release, which is 3.7 at this point in 2018. However, there has been some errors using Tensorflow with this version. Therefore, ```Python 3.6``` is advisable.

```
$ brew install https://raw.githubusercontent.com/Homebrew/homebrew-core/f2a764ef944b1080be64bd88dca9a1d80130c558/Formula/python.rb
```

Check the version of ```Python 3``` we just installed:

```
$ python3 --version
Python 3.6.5
```

Reference:

- https://wsvincent.com/install-python3-mac/
- https://stackoverflow.com/questions/51125013/how-can-i-install-a-previous-version-of-python-3-in-macos-using-homebrew/51125014#51125014

## 4. Anaconda

We are going to install Tensorflow with Virtualenv. Virtualenv is a virtual Python environment isolated from other Python development, incapable of interfering with or being affected by other Python programs on the same machine. During the Virtualenv installation process, you will install not only TensorFlow but also all the packages that TensorFlow requires.

In Anaconda, you may use conda to create such a virtual environment.

Download the most recent release Anaconda from ```https://repo.anaconda.com/archive/Anaconda3-5.3.1-MacOSX-x86_64.sh```

Install Anaconda
```
$ bash Anaconda3-5.3.1-MacOSX-x86_64.sh
```

Note there will be some questions during the installing process we need to take care of. You may also need to modify your ```PATH```:

```
export PATH=$HOME/anaconda/bin:$PATH
```
to your ```.bash_profile```



Create a conda environment named ```tensorflow```:

```
(start new terminal)
$ conda create -n tensorflow pip python=3.6
```

Activate the conda environment ```tensorflow```:

```
$ source activate tensorflow
(tensorflow)$  # Your prompt should change
```

Install TensorFlow inside your conda environment using the command below. Here, I aim to install CPU-only Tensorflow version 1.8. Note, as of version 1.2, TensorFlow no longer provides GPU support on Mac OS X.

```
(tensorflow)$ pip install --ignore-installed --upgrade \
 https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.8.0-py3-none-any.whl
```

Validate the installed tensorflow:

```
(tensorflow)$ python3
(enter the following command)
>>> import tensorflow as tf
>>> hello = tf.constant('Hello, TensorFlow!')
>>> sess = tf.Session()
>>> print(sess.run(hello))
Hello, TensorFlow!
```
