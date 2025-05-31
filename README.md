# findr
FINDR: Flow-field Inference from Neural Data using deep Recurrent networks

# Installation
Run the commands below to install FINDR:

```
$ git clone https://github.com/Brody-Lab/findr
$ module load anaconda/2024.10
$ conda create --name findr python=3.12
$ conda activate findr
$ cd findr
$ pip install -e .
```

# Data format
The data needs to be stored as an `.npz` file that contains the following keyword arguments:

`spikes`: contains a 3-d array (# of trials x maximum trial length x # of neurons) of spike counts for each time bin.

`externalinputs`: contains a 3-d array (# of trials x maximum trial length x input stimulus dimension) where the input stimulus dimension can be an integer greater than or equal to 1. The stimulus values themselves can be floating point numbers or integers.

`lengths`: contains a 1-d array (# of trials) of the length of each trial (in the unit of time bins).

`times`: contains a 1-d array (# of trials) of the timestamp of onset of each trial.

# Training FINDR

Run the commands below to run FINDR:

```
$ module load anaconda/2024.10
$ conda activate findr
$ python main.py --datapath=$datafilepath --workdir=$analysispath
```

Make sure that the `$datafilepath` correctly specifies the location of the data file to fit (in `.npz` format). The `$analysispath` is where the trained FINDR parameters are stored.

It should take a few hours on a single A100 GPU to finish training.

# Example analyses
There are example Jupyter notebooks under the `notebooks` folder. The `plot_example_vector_fields.ipynb` notebook demonstrates how to plot flow fields (or the velocity vector fields) for an example dataset.

# Citation

Kim, T.D., Luo, T.Z., Can, T., Krishnamurthy, K., Pillow, J.W., Brody, C.D. (2025). Flow-field inference from neural data using deep recurrent networks. *Proceedings of the 42nd International Conference on Machine Learning (ICML)*.

```bibtex
@article{kim2025findr,
    author={Timothy Doyeon Kim and Thomas Zhihao Luo and Tankut Can and Kamesh Krishnamurthy and Jonathan W. Pillow and Carlos D. Brody},
    title={Flow-field inference from neural data using deep recurrent networks},
    year={2025},
    journal={Proceedings of the 42nd International Conference on Machine Learning (ICML)}
}
```