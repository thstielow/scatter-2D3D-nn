# Reconstruction of nanoscale particles from single-shot wide-angle FEL diffraction patterns with physics-informed neural networks
The repository `scatter-2d3d-nn` provides the code used in the equally named [paper](https://arxiv.org/abs/2101.09136) for the reconstruction of three dimensinal object densities of nanoclusters from single-shot wide angle diffraction patterns. For details on the physical beckground refer to the arXiv preprint [2101.09136](https://arxiv.org/abs/2101.09136).

## Usage
The [main notebook](https://github.com/thstielow/scatter-2D3D-nn/blob/main/main.ipynb) demonstrates the physics informed training of the scatter reconstruction nn on a sample portion of the original dataset. The neural network constructor can be found in [scatter_rec_model.py](https://github.com/thstielow/scatter-2D3D-nn/blob/main/scatter_rec_model.py), while [physics_loss.py](https://github.com/thstielow/scatter-2D3D-nn/blob/main/physics_loss.py) implements the physics informed loss function.

The notebook further contains demonstrations of the predictive capacity upon loading the weights  from [cluster2Dto3Dweights.h5 ](https://github.com/thstielow/scatter-2D3D-nn/blob/main/cluster2Dto3Dweights.h5) which are the same weights which were used throuhout the [paper](https://arxiv.org/abs/2101.09136).

## Data
Sample data for both training and validation are provided in the [datasets](https://github.com/thstielow/scatter-2D3D-nn/tree/main/datasets) folder. The original dataset contains 140k image pairs, which were split at a ratio of 0.2 into a training and validation set. The sample sets are the first 1000 and the last 100 image pairs for the training and validation set respectively. The full dataset is available from the corresponding author upon request.

The test set constructed from 1000 chopped base solids as described in the [paper](https://arxiv.org/abs/2101.09136) is provided in full.

The experiment data were extracted from the [paper by Barke et al.](https://doi.org/10.1038/ncomms7187) and rescaled to fit the required dimensions and resolution. The use is permitted by the [Creative Commons CC-BY 4.0 license](http://creativecommons.org/licenses/by/4.0/).


## Requirements
- Tensorflow >= 2.3
