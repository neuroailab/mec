# Explaining heterogeneity in medial entorhinal cortex with task-driven neural networks

**Aran Nayebi, Alexander Attinger, Malcolm G. Campbell, Kiah Hardcastle, Isabel I.C. Low, Caitlin S. Mallory, Gabriel C. Mel, Ben Sorscher, Alex H. Williams, Surya Ganguli, Lisa M. Giocomo, Daniel L.K. Yamins**

*35th Conference on Neural Information Processing Systems (NeurIPS 2021)*

[Preprint](https://www.biorxiv.org/content/10.1101/2021.10.30.466617)

## Getting started

First clone this repo, then install dependencies
```
pip install -r requirements.txt
```
We recommend Python 3.6 if you run the above requirements file.

## Training code

To train the UGRNN-ReLU-Place Cell path integrator, use the following command:

```python run_trainer.py --gpu=[GPU_ID] --rnn_type="UGRNN" --activation="relu" --save_dir=[MODEL_SAVE_DIR] --run_ID="ugrnn_relu"```

To train the UGRNN-ReLU-Place Cell path integrator with cue inputs, use the following command:

```python run_trainer.py --gpu=[GPU_ID] --rnn_type="CueUGRNN" --activation="relu" --train_with_cues=True --cue_prob=0.5 --save_dir=[MODEL_SAVE_DIR] --run_ID="cueugrnn_relu"```

To train the UGRNN-ReLU-Place Cell path integrator with rewards, use the following command:

```python run_trainer.py --gpu=[GPU_ID] --rnn_type="UGRNN" --activation="relu" --reward_zone_size=0.2 --save_dir=[MODEL_SAVE_DIR] --run_ID="ugrnn_relu_reward"```

## Pre-trained models

To get the saved checkpoints of the above three models, simply run this script

```
./get_checkpoints.sh
```
This will save them to the current directory.

You can then load each of these models as a TensorFlow object by running the following commands:

UGRNN-ReLU-Place Cell path integrator:
```
from mec.models.utils import load_trained_model
ugrnn_relu_model = load_trained_model(rnn_type="UGRNN", activation="relu",
                                load_dir="[DIR_PATH]/mecmodels/",
                                run_ID="UGRNN_relu",
                               ckpt_file="ckpt-101")
```

UGRNN-ReLU-Place Cell path integrator with cue inputs:
```
from mec.models.utils import load_trained_model
from mec.core.constants import CUE_2D_INPUT_KWARGS
cue_ugrnn_relu_model = load_trained_model(rnn_type="CueUGRNN", activation="relu",
                                cue_2d_input_kwargs=CUE_2D_INPUT_KWARGS,
                                load_dir="[DIR_PATH]/mecmodels/",
                                run_ID="CueUGRNN_relu",
                               ckpt_file="ckpt-101")
```

UGRNN-ReLU-Place Cell path integrator with rewards:
```
from mec.models.utils import load_trained_model
from mec.core.constants import REWARD_KWARGS
reward_ugrnn_relu_model = load_trained_model(rnn_type="UGRNN", activation="relu",
                                load_dir="[DIR_PATH]/mecmodels/",
                                run_ID="UGRNN_relu_reward",
                                ckpt_file="ckpt-101",
                                **REWARD_KWARGS)
```

## Computing inter-animal consistency and model neural fits

### Inter-animal consistency
The `get_fits()` function is the main entry point for computing inter-animal consistency and model neural fits.
If you have some data, called `dataset`, from multiple animals for a given (or multiple) arena sizes, ensure first that `dataset` is a dictionary in the following format:
The outermost keys of `dataset` are the arena names, then the keys of `dataset[arena_name]` are the animal names, and then the keys of `dataset[arena_name][animal_name]`
consist of two elements: `resp` and `cell_ids`.
`dataset[arena_name][animal_name]["resp"]` is the binned response of the animal in that arena, typically of shape `(num_x_bins, num_y_bins, num_cells)`, binned into 5 cm bins.
`dataset[arena_name][animal_name]["cell_ids"]` is a NumPy array of length `num_cells` consisting of the unique ids assigned to each cell (this is completely up to you, but one simple convention might be `[animal_name]_cell0`, `[animal_name]_cell1`, and so forth).

Here is how to then compute the inter-animal consistency on 20% of the bins with 10 train-test splits with Ridge (alpha=1) regression:
```
from mec.neural_fits.comparisons import get_fits
results = get_fits(
    dataset=dataset,
    arena_sizes=list(dataset.keys()),
    interanimal_mode="holdout",
    map_type="sklinear",
    map_kwargs={},
    train_frac=0.2,
)
```
By default, we compute the inter-animal consistency where the source animal is the concatenation of all animals distinct from the target animal (set by `interanimal_mode="holdout"`), which was used in the paper to account for the fact that there are not too many cells for any single animal.
However, if you would like to compute it on a pairwise basis, set `interanimal_mode="pairwise"`, though this tends to give lower inter-animal consistencies.

There are three types of maps, `map_type="pls"` for Partial Least Squares (PLS) regression; `map_type="sklinear"` for Lasso, Ridge, and ElasticNet regression (along with any other `sklearn` linear models); and finally `map_type="corr"` which corresponds to the One-to-One mapping where we find the neuron in the source animal most correlated with that target unit on the training bins.

If you would rather test other types of generalization other than randomly chosen bins (such as training on the left half of the arena and testing on the right half), pass in a string instead.
The currently supported string-based generalizations are in the `generate_train_test_splits()` function of `mec.neural_fits.utils`.

If you want to then print summary statistics (e.g. median) of the inter-animal consistency across all units of all animals in a given arena size, use the `unit_concat()` function:
```
import numpy as np
from mec.neural_fits.utils import unit_concat
np.median(unit_concat(results, arena_size=ARENA_SIZE, inner_key="corr"))
```
Finally, while we found Ridge (alpha=1) regression to produce similar results, if you have the computational resources and would like to run your own ElasticNet CV regression per cell on your own data, we have included the range of alpha and l1 ratio values used in our grid search in `ALPHA_RNG` and `L1_RATIO_RNG` in  `mec.core.constants`.
If you prefer a smaller range due to computational constraints, we recommend using instead `ALPHA_RNG_SHORT` and `L1_RATIO_RNG_SHORT`.
We recommend two-fold cross-validation on 20% of the bins (so two runs of 10\% each) per cell and per train-test split, for a given choice of hyperparameters.
Once you have the settings for each cell determined, then use `map_type="sklinear"` and pass in `map_kwargs_per_cell` when calling `get_fits()` rather than `map_kwargs`,  and include in each cell's kwargs the entry `"regression_type": "ElasticNet"`.

### Model comparisons
To compare a model, take the model object returned by `load_trained_model()` and the environment configuration, specified through `configure_options()`.
Note that the environment configuration should be the same size as the environment the neural data is collected in, as specified by `box_width` and `box_height` (in meters).

Here is an example where we compare the UGRNN-ReLU-Place Cell model with Ridge (alpha=1) regression, from the `"g"` (second) layer of the model (the best layer compared to the other two layers `"pre_g"` and `"dc"`), to neural data collected from a 2.2m x 2.2m arena.
```
from mec.models.utils import load_trained_model, configure_options
ugrnn_relu_model = load_trained_model(rnn_type="UGRNN", activation="relu",
                                load_dir="[DIR_PATH]/mecmodels/",
                                run_ID="UGRNN_relu",
                               ckpt_file="ckpt-101")

eval_cfg = configure_options(box_width=2.2, box_height=2.2)

model_results = get_fits(
    model=ugrnn_relu_model,
    cfg=eval_cfg,
    model_pred_layer="g",
    dataset=dataset,
    arena_sizes=[2.2],
    map_type="sklinear",
    map_kwargs={},
    train_frac=0.2,
)
```
The bins used to compute the model rate maps will be printed out for convenience, please ensure they match what you used for computing the data rate maps!
Note by default, the minimum and maximum height and width of the arena are set to be [-h/2, h/2] and [-w/2, w/2], respectively.

If you prefer a different environment extent, pass in `min_x`, `max_x`, `min_y`, and `max_y` into `configure_options()` in units of meters.
If you prefer a different number of bins, pass in `nbins` to `get_fits()`, and if you prefer to use a different number of centimeters per bin (default is 5 cm), pass in `bin_cm`.

Finally, if you prefer to pass in a custom model binned response, of shape `(num_x_bins, num_y_bins, num_units)`, you can do that via the argument `model_resp`, instead of passing in the `model` object and its environment configuration.

## Cite

If you used this codebase for your research, please consider citing our paper:
```
@inproceedings{mectask2021,
    title={Explaining heterogeneity in medial entorhinal cortex with task-driven neural networks},
    author={Nayebi, Aran and Attinger, Alexander and Campbell, Malcolm G and Hardcastle, Kiah and Low, Isabel IC and Mallory, Caitlin S and Mel, Gabriel C and Sorscher, Ben and Williams, Alex H and Ganguli, Surya and Giocomo, Lisa M and Yamins, Daniel LK},
    booktitle={The 35th Conference on Neural Information Processing Systems (NeurIPS 2021)},
    url={https://www.biorxiv.org/content/10.1101/2021.10.30.466617},
    year={2021}
}
```

## Contact

If you have any questions or encounter issues, either submit a Github issue here or email `anayebi@stanford.edu`.
