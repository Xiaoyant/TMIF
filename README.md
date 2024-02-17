# TMIF
A Transformer-based Multi-modal Integration Framework (TMIF) for fusulinid fossils classification

### **The TMIF diagram**

![TMIF](https://github.com/xiaoyantxx/TMIF/assets/154036426/7682d6df-b88c-4105-b214-d6ea325f320d)

From the piper: Enhanced Taxonomic Identification of Fusulinid Fossils through Image-Text Integration Using Transformer

### What is this repository for?

TMIF is a framework that uses deep learning for model training and classification of fusulinid fossils.

### How do I get set up?

Install python 3.8 in the anaconda virtual environment on the Ubuntu operating system. Follow the requirements.txt file to configure the corresponding version of the package.

### Usage

1) Description of the document:

   data: storing dataset.

   dataManagement: it is responsible for separating the "class|image|text|" of a multi-modal dataset, processing and loading the dataset.

   multiManagement: it is responsible for obtaining the text vocabulary, and converting the images and text descriptions to encoding.

   models: the network file for TMIF. CrossTransIngration.py describes the overall network framework, and other files are called by it.

   runs: it holds the model trained by train.py.

   train.py: for training optimal models.

   test.py: for testing.

   requirements: includes a number of dependent packages.

2) Download the optimal training model trained by the authors. Click on the link: https://pan.baidu.com/s/16-hUw3IMuOb0gf0dr9AuQg?pwd=3f74. The extraction code is 3f74. Download the model and place it in the /runs/weights folder. 

3) Deploy the environment in IDE or terminal. Click test.py, set the weights file and the path to the multimodal dataset (already set). Click run to output the class predicted by the model.

4) The output prediction images are saved in the /runs/weights file.

### Who do I talk to?

Fukai Zhang, Henan Polytechnic University

Email: zhangfukai@hpu.edu.cn
