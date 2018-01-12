# segmentation_sol

## Installation

- Install `conda` (https://conda.io/miniconda.html) or update it
```
conda update conda
```
- Tp create a new environment `segsol`, run the command:
```
conda create -n segsol python=3.6
```
- To activate this environment, run the command:
```
source activate segsol
```
- To install `pytorch` package (http://pytorch.org/), run the command:
```
conda install pytorch torchvision -c pytorch 
```
- To install other package, run the command: 
```
pip install tqdm
```
- To clone this repository, run the command: 
```
git clone https://github.com/durandtibo/segmentation_sol.git
```

## Evaluation


From the folder `segmentation_sol`, to predict the mask of one image, run the command: 
```
python -m segmentation.main --image data/test.JPG --max_size 1000 --output_dir outputs
```


From the folder `segmentation_sol`, to predict the mask of all the images in a directory, run the command: 
```
python -m segmentation.main --image data --max_size 1000 --output_dir outputs
```

List of options
- `image`: the filename or the directory of the image
- `checkpoint_dir`: the directory where the checkpoint is
- `output_dir`: the directory where to save the results 
- `max_size`: the size of the smaller edge of the image