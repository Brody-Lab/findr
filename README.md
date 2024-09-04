# findr
FINDR: Flow-field Inference from Neural Data using deep Recurrent networks

# Installation
## Princeton Della
Run the commands below to install flax on Della:

```
$ ssh <YourNetID>@della-gpu.princeton.edu
$ module load anaconda3/2022.5
$ conda create --name findr python=3.9 matplotlib ipykernel scikit-learn -c conda-forge
$ conda activate findr
$ pip install jaxlib[cuda11_cudnn82]==0.4.4 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
$ pip install jax==0.4.4
$ pip install flax==0.6.4
$ pip install ml_collections==0.1.1
```

Then clone https://github.com/Brody-Lab/findr to `/home/<YourNetID>/` on Della.

# Training an example session
Go to `run_findr_array.sh` inside `findr`.

```
$ ssh <YourNetID>@della-gpu.princeton.edu
$ cd findr
$ nano run_findr_array.sh
```

Make sure that every `tdkim` in this file is changed to `<YourNetID>`. What this shell script does is to identify all data files (stored in `.npz` format) in `datafolderpath` and run the `session_id`-th data file in `datafolderpath`. The first data file starts with `session_id=0`, not `session_id=1`. To get the data files in `datafolderpath="/scratch/gpfs/tdkim/findr/npx_luo/manuscript2023a/recordingsessions/2024_04_09"`, go to `mnt/cup/labs/brody/tdkim/manuscript2023a/recordingsessions/2024_04_09` and move the `.npz` files to any folder you create under `/scratch/gpfs/<YourNetID>/`. Make sure that the `datafolderpath`, `analysisfolderpath` and `session_id` are correctly specified. 

Run `run_findr_array.sh` using
```
$ sbatch run_findr_array.sh
```

This is it! It should take a few hours for all runs to finish.

# Training FINDR on your data
The data needs to be stored as an `.npz` file that contains the following keyword arguments:

`spikes`: contains a 3-d array (# of trials x maximum trial length x # of neurons) of spike counts for each time bin.

`choices` (optional): contains a 1-d array (# of trials) of the animal's binary choice if the task is a two-alternative forced choice task (leftward = 0, rightward = 1). FINDR does not make use of `choices` during training. Currently, `choices` are used only during post-modeling analysis.

`externalinputs`: contains a 3-d array (# of trials x maximum trial length x input stimulus dimension) where the input stimulus dimension can be an integer greater than or equal to 1. The stimulus values themselves can be floating point numbers or integers. For the Poisson Clicks task, input stimulus dimension is 2, with [0, 0] indicating the time bin with no clicks, [1, 0] indicating the time bin with a left click, and [0, 1] indicating the time bin with a right click.

`lengths`: contains a 1-d array (# of trials) of the length of each trial (in the unit of time bins).

`times`: contains a 1-d array (# of trials) of the timestamp of onset of each trial.

# References
1. Kim, T.D., Luo, T.Z., Can, T., Krishnamurthy, K., Pillow, J.W., Brody, C.D. (2023). Flow-field inference from neural data using deep recurrent networks. bioRxiv.
2. Luo, T.Z.\*, Kim, T.D.\*, Gupta D., Bondy, A.G., Kopec, C.D., Elliot, V.A., DePasquale, B., Brody, C.D. (2023). Transitions in dynamical regime and neural mode underlie perceptual decision-making. bioRxiv.
