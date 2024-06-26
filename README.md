This is the code of "MRF<sup>3</sup>Net: Infrared Small Target Detection Using Multi-Receptive Field Perception and Effective Feature Fusion". Our work has been accepted by IEEE Transactions on Geoscience and Remote Sensing (TGRS). Please refer to our [paper](https://ieeexplore.ieee.org/document/10562332) for more details.

To train the model:
```python
CUDA_VISIBLE_DEVICES=0 python main.py --Task train --save_dir logs/ --dataset_path path_to_dataset --img_size 256
```

To validate the model:
```python
CUDA_VISIBLE_DEVICES=0 python main.py --Task validation --weights path_to_your_weight --dataset_path path_to_dataset --img_size 256
```
