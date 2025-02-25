# Pruning is all you need

Repository for the paper "Smaller is Better: Enhancing Tansparency in Vehicle AI Systems via Pruning", submitted on 3rd USENIX Symposium on Vehicle Security and Privacy (VehicleSec).

This paper provides our implementation of building efficient and interpretable models using various deep learning model Pruning strategies.

## Requirement
- kornia (for applying filters to feature-maps)
- cleverhans (for adversarial training)
- captum (for saliency maps)
- quantus (for computing several metrics for saliency maps)
- Ranger Optimizer (for optimizing ResNet model during training)


## Structure
    - Explanations: This folder consists of all the quanlitative explanations produced by the VGG and RESNET models.
    - GTSRB: This folder consists of explanation and roadplots for German Traffic Sign Recommendation Benchmark dataset.
    - datasets: This folder consists subset LISA and Subset LISA dataset.
    - roadplots: This program file consists of roadplots for VGG and ResNet models.
    - lisa_subset.py: Makes dataset from image tensor files for LISA Subset.
    - make_subset.py:Makes subset tensor files from LISA tensor files.
    - metrics.ipynb: Implementation of helper functions related to explanation metrics.
    - models.ipynb: Implementation of VGG and ResNet model.
    - utils.ipynb: Helper functions.
    - resnet_lisa_saliency: Implementation of qualitative and quantitative metrics for ResNet.
    - resnet_lisa_train: Code to train the models for ResNet.
    - vgg_lisa_saliency:Implementation of qualitative and quantitative metrics for VGG.
    - vgg_lisa_train:Code to train the models for VGG.
    
    
    
