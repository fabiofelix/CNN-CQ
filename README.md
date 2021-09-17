# A classification and quantification approach to generate features in soundscape ecology using neural networks

This repo contains a code implementation of the loss function proposed by the [paper](https://doi.org/10.1007/s00521-021-06501-w) published on [Neural Computing and Applications](https://www.springer.com/journal/521/). It created a custom loss function that combines cross-entropy with [quantification](https://doi.org/10.1007/s13748-016-0103-3) to train a simple CNN and the ResNet-50 to classify an audio set compounded by [natural sounds](https://doi.org/10.1007/s10980-011-9600-8).  

## Prerequisites

To run the code, you will need to install:

* [Python 3.5.2](https://www.python.org/downloads/release/python-352/)

* [tensorflow-gpu 1.10.0](https://pypi.org/project/tensorflow-gpu/1.10.0/)

> obs.: It ran on Cuda 9.0 and cuDNN 7.0.5.15

* [Keras 2.2.5](https://pypi.org/project/keras/2.2.5/)

* [pandas 0.24.2](https://pypi.org/project/pandas/0.24.2/)

* [scikit-learn 0.22.1](https://pypi.org/project/scikit-learn/0.22.1/)

## Description

We created two python files, and the main.py has the main functions we used: the architectures tested (my_CNN2D and my_ResNet50) and the custom loss (count_CC_loss). We do not know if that is the better way to implement the loss function, but it works. 

The utils.py has some auxiliary functions used to train/apply the models. We coded a simple generator function, but you can use your generators.

The main code has the following parameters:

```python
  -a  Action to be executed, to train a model (train) or to appply some pretrained model (apply) 
  -s  Source directory to load spectrogram images.
  -t  Target directory to save/load model. Default = current directory
  -m  Model index (6) CNN (18) ResNet
  -l  Quantity of labels that the model needs to classify
  -e  Quantity of epochs. Default = 100
  -b  Batch size used for training, validation and test. Default = 80
  -quant [value] Use the custom loss function with quantification. (no value after the parameter or 1) 
                 first weighting case, (2) second weighting case, (3) third weighting case
  -eval Generate model evaluation. Default = False.
```

## Running

To **train** with your dataset, the generator looks for two directories inside the source path: train and validation. Inside these directories, put your spectrogram images and a CSV file with two columns: file and label related to your spectrograms.

To **apply** our pre-trained models to your data, set target *-t* parameter with a directory that contains a model (one of the files inside our *model* directory) and set source *-s* parameter with your spectrograms directory. Code will generate a CSV with the predicted labels in the current path, considering the case 12-class (described in the paper).

If you want other class scenarios (bird-class, anuran-class, or 2-class), just point to the specific model (see our *model* directory) and change the function utils.decode_labels, removing the commentary of your desired case.

Besides, you can use a ground truth to evaluate the models. Put a CSV file with two columns (file and expected labels) inside the source path and pass *-eval* parametter.

## Examples

```python
  python main.py -a apply -l 12 -m 6 -b 20 -quant -eval -s /home/user/Desktop/data/test/ -t /home/user/Desktop/model
```

```python
  python main.py -a train -l 12 -m 18 -e 50 -b 20 -quant -s /home/user/Desktop/data/ -t /home/user/Desktop/model
```  

## Data used

The complete database has been collected by the [LEEC lab](https://github.com/LEEClab) and subsets were used in other papers, such as [[1]](https://doi.org/10.1016/j.ecolind.2020.107050), [[2]](https://doi.org/10.1016/j.ecolind.2020.107316), and [[3]](https://doi.org/10.3390/info12070265). Our subset is labeled with animal species and will be available on the [lab website](https://github.com/LEEClab) as soon as possible.



## Contact

* [Fábio Felix Dias](https://scholar.google.com.br/citations?hl=pt-BR&user=uQ_qg2MAAAAJ) - e-mail: <f_diasfabio@usp.br>

* [Moacir Antonelli Ponti](https://scholar.google.com.br/citations?user=ZxQDyNcAAAAJ&hl=pt-BR&oi=sra) - e-mail: <moacir@icmc.usp.br>

* [Rosane Minghim](https://scholar.google.com.br/citations?user=TodwpSwAAAAJ&hl=pt-BR&oi=ao) - e-mail: <rosane.minghim@ucc.ie>  
