# DNNScaler

To improve the inference throughput of DNNs deployed on GPU accelerators, two common approaches are employed: Batching and Multi-Tenancy. Our preliminary experiments show that the effect of these approaches on the throughput depends on the DNN architecture. Taking this observation into account, we design and implementDNNScalerwhich aims to maximize the throughput of interactive AI-powered services while meeting their latency requirements.DNNScaler first detects the suitable approach (Batch-ing or Multi-Tenancy) that would be most beneficial for a DNNN regarding throughput improvement. Then, it adjusts the control knob of the detected approach (batch size for Batch-ing and number of co-located instances for Multi-Tenancy)to maintain the latency while increasing the throughput. 


## Requirements
* TensorFlow GPU (TF V1)
* CUDA
* cuDNN
* Matlab
* TFOCS (http://cvxr.com/tfocs/)

## DNN Models
We have chosen sixteen DNNs with different characteristics such as size and computational complexity to show the applicability of BatchDVFS on a wide variety of DNNs. The DNNs have been selected from [TensorFlow-Slim image classification model library](https://github.com/tensorflow/models/tree/master/research/slim). We have followed the instructions provided in the aforementioned library to generate the frozen graphs of the pre-trained models. **You can download the frozen graphs from this [link](https://drive.google.com/file/d/1QJFxeoO_gmZiK-vzM75OQnA0XjL5ZL9P/view?usp=sharing)**

## Datasets
We have two image datasets, one from [ImageNet](http://www.image-net.org/) which is a popular dataset that is widely used in other works, and the other one is [CalTech 256](http://www.vision.caltech.edu/Image_Datasets/Caltech256/) which is collected by researchers from the California Institute of Technology.

## Usage

The Batching and Multi-Tenancy folders contain the Python files for DNNScaler. The Clipper folder contains Python files for the Clipper approach. The Python files need to be copied into a folder that contains the frozen graphs and the datasets. Then, by executing the commands in the .sh files, the experiments can be conducted.

The MatrixCompletion folder also contains the files for MC estimation. The TFOCS needs to be downloaded, and its path should be updated in the .m file.
