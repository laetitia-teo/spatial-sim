# SpatialSim Howto

Hi, and welcome to the codebase of the paper SpatialSim, Recognizing Spatial Configurations of objects with Graph Neural Networks. This file will explain how to use the datasets described in the paper, and reproduce the experiments in the main text.

## Getting the data

After cloning the repository, get the data files from this [link](https://drive.google.com/file/d/1uMonAZTQoHc4e_0bS24kExWw9Fq6_LdO/view?usp=sharing), and extract it in the repo.

## Replicating experiments

Experiment replication is done in two steps: first, generate the configuration file for the wanted experiment; second, launch the created runfile.

### Identification

For Identification, please use the command line command

```
python generate_config.py --mode simple
```

The default datasets used are IDS_3 to IDS_8, containing 3 to 8 objects. For changing this behavior, one can specify the minimum number of objects and maximum number of objects, respectively, by using the `-No` (resp. `-Nm`) flag. Thus, the above command is equivalent to

```
python generate_config.py --mode simple -No 3 -Nm 8
```

After typing the command, a number is printed in the terminal; it corresponds to the identifying index of the configuration file and of the experiment. The configuration file is stored in the `configs` folder under `config<index>`. To launch the experiment, please type (if using bash):

```
bash launch<index>.sh
```

The results are stored in the `experimental_results` folder, under `expe<index>`. Inside the folder, one can find the log file, and a folder for each of the trained models, inside which there is a folder `data` for the train and test accuracies and a folder `models` where the weights for the different seeds of the models, at the end of training, are stored.

### Discrimination

For Discrimination, the process is similar. To replicate the experiments in the paper, please type:

```
python generate_configs.py --mode double
```

As in the previous case, the `-No` and `-Nm` flags allow to control the minimum and maximum numbers of objects. The different allowed combinations of these two flags are `-No 3 -Nm 8` (default), `-No 9 -Nm 20`, and `-No 21 -Nm 30`.

Once the configuration file is generated (its index is printed in the terminal), please execute 

```
bash launch<index>.sh
```

to launch the experiment.

### Getting the metrics

To access the results of training (test accuracies), please enter the python interpreter, and then execute:

```python
>>> from run_utils import *
>>> model_metrics(<index>)
```

## Using SpatialSim data

We provide utility scripts for handling the data. To create pytorch `Datasets` and `Dataloaders`, use the following commands:

```python
>>> from run_utils import load_ds, load_dl
>>> data_path = 'data/double/CDS_3_8'  # example dataset
>>> dataset = load_ds(data_path)
>>> dataloader = load_dl(data_path)
```

### Using image data

Image data for the different datasets car be rendered based on the provided data. To do so, please use the following script:

```bash
$ python make_imgs.py --task [TASK] --n_proc [N_PROC]
```

`[TASK]` is one of `'simple'` (for Identification) or `'double'` (for Discrimination). `[N_PROC]` is the number of processes to use for image generation, defaults to 4.

After image generation, load the image data as so:

```python
>>> from run_utils import load_ds, load_dl
>>> data_path = 'data/double/CDS_3_8'  # data path stays the same
>>> dataset = load_ds(data_path, use_images=True)
>>> dataloader = load_dl(data_path, use_images=True)
```

The datasets and dataloaders will then output images instead of sets of points.
