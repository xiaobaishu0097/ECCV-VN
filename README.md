# Learning Object Relation Graph and Tentative Policy for Visual Navigation (Updating)

## Overview

## Dependencies

- python 3.6+
- python libraries
  - pytorch 1.4+ & tochvision
  - ai2thor
  - networkx
  - numpy
  - scipy
  - tensorboardX

Install the necessary packages. If you are using pip then simply run `pip install -r requirements.txt`.

## Data

Your `~/Data/` folder should look like this
```
Data
└── AI2thor_offline_data/
    ├── FloorPlan1
    │   ├── resnet18_featuremap.hdf5
    │   ├── graph.json
    │   ├── visible_object_map_1.5.json
    │   ├── det_feature_categories.hdf5
    │   ├── grid.json
    │   └── optimal_action.json
    ├── FloorPlan2
    └── ...
```
You can download the dataset used in our paper [here](https://drive.google.com/file/d/1kvYvutjqc6SLEO65yQjo8AuU85voT5sC/view?usp=sharing) and test data [here](https://drive.google.com/file/d/1ud_2OQfFFJOufdz_KYBdgJesEGkwy7Nj/view?usp=sharing).

### Training 

```python
python main.py --model GraphModel --worker 12 --gpu-ids 0 1 --title a3c_graph
```

### Evaluating

```python
python full_eval.py --model GraphModel --results_json a3c_graph_results.json --gpu-ids 0 1 --title a3c_graph
```
