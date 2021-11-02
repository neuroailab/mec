from scipy import stats
import os, copy
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from mec.models.model import (
    UGRNN,
    CueUGRNN,
)
from mec.models.place_cells import PlaceCells
from mec.models.trajectory_generator import TrajectoryGenerator


def set_env_dims(options):
    if (not hasattr(options, "min_x")) or (not hasattr(options, "max_x")):
        assert hasattr(options, "box_width") and (options.box_width is not None)
        options.min_x = -options.box_width / 2.0
        options.max_x = options.box_width / 2.0
    else:
        assert (options.min_x is not None) and (options.max_x is not None)

    if (not hasattr(options, "min_y")) or (not hasattr(options, "max_y")):
        assert hasattr(options, "box_height") and (options.box_height is not None)
        options.min_y = -options.box_height / 2.0
        options.max_y = options.box_height / 2.0
    else:
        assert (options.min_y is not None) and (options.max_y is not None)


def compute_ratemaps(
    model,
    options,
    n_eval_seq=None,
    trajectory_seed=0,
):
    """Compute spatial firing fields"""

    assert hasattr(model, "g")
    assert hasattr(model, "dc")
    if hasattr(model, "pre_g"):
        pre_g = np.zeros(
            [n_eval_seq, options.batch_size * options.sequence_length, options.Ng]
        )
    dc = np.zeros(
        [n_eval_seq, options.batch_size * options.sequence_length, options.Np]
    )

    trajectory_generator = TrajectoryGenerator(
        options, PlaceCells(options), trajectory_seed=trajectory_seed
    )

    if not n_eval_seq:
        n_eval_seq = 1000 // options.sequence_length

    g = np.zeros([n_eval_seq, options.batch_size * options.sequence_length, options.Ng])
    p = np.zeros([n_eval_seq, options.batch_size * options.sequence_length, options.Np])
    pos = np.zeros([n_eval_seq, options.batch_size * options.sequence_length, 2])
    inp = []

    for index in tqdm(range(n_eval_seq), leave=False, desc="Computing ratemaps"):
        inputs, pos_batch, p_batch = trajectory_generator.get_batch()
        # the element of the input tuple is velocity or cues depending on the options passed in
        relevant_inp = inputs[0]
        # batch x sequence length x dimensionality
        assert len(relevant_inp.shape) == 3
        inp_dim = relevant_inp.shape[-1]
        # batch * sequence length x dimensionality
        inp.append(np.reshape(relevant_inp, (-1, inp_dim)))
        if hasattr(model, "g"):  # means it is an RNN
            if hasattr(model, "pre_g"):
                pre_g_batch = model.pre_g(inputs)
                assert pre_g_batch.shape[-1] == options.Ng
                pre_g_batch = np.reshape(pre_g_batch, (-1, options.Ng))
                pre_g[index] = pre_g_batch

            g_batch = model.g(inputs)
            assert g_batch.shape[-1] == options.Ng
            g_batch = np.reshape(g_batch, (-1, options.Ng))
            g[index] = g_batch

            dc_batch = model.dc(inputs)
            assert dc_batch.shape[-1] == options.Np
            dc_batch = np.reshape(model.dc(inputs), (-1, options.Np))
            dc[index] = dc_batch

        p_batch = np.reshape(p_batch, (-1, options.Np))
        p[index] = p_batch

        pos_batch = np.reshape(pos_batch, (-1, 2))
        pos[index] = pos_batch

    g = g.reshape((-1, options.Ng))
    dc = dc.reshape((-1, options.Np))
    g_dict = {"g": g, "dc": dc}
    if hasattr(model, "pre_g"):
        pre_g = pre_g.reshape((-1, options.Ng))
        g_dict["pre_g"] = pre_g

    p = p.reshape((-1, options.Np))
    pos = pos.reshape((-1, 2))
    inp = np.stack(inp, axis=0)
    inp = inp.reshape((-1, inp.shape[-1]))

    return g_dict, p, pos, inp


def get_model_activations(
    model,
    cfg,
    nbins=None,
    bin_cm=5.0,
    n_eval_seq=100,
    trajectory_seed=0,
    model_pred_layer=None,
):
    """Returns (num_x_bins, num_y_bins, num_units) model activations for a given arena size"""
    print(f"Using this cfg to compute model activations: {vars(cfg)}")
    if nbins is None:
        assert cfg.box_width == cfg.box_height
        # convert meters to cm and then bin into 5 cm bins
        nbins = (int)((cfg.box_width * 100) / ((float)(bin_cm)))
    print(f"Binning into {nbins} bins")
    arena_x_bins = np.linspace(cfg.min_x, cfg.max_x, nbins + 1, endpoint=True)
    arena_y_bins = np.linspace(cfg.min_y, cfg.max_y, nbins + 1, endpoint=True)
    print(
        f"Using these X bins for model activations: {arena_x_bins}, make sure they line up with the bins used for the data!"
    )
    print(
        f"Using these Y bins for model activations: {arena_y_bins}, make sure they line up with the bins used for the data!"
    )

    # get model activations
    g_dict, p, pos, inp = compute_ratemaps(
        model,
        options=cfg,
        n_eval_seq=n_eval_seq,
        trajectory_seed=trajectory_seed,
    )

    # return all layers of an rnn simultaneously
    assert isinstance(g_dict, dict)
    model_activations = {}
    for k, v in g_dict.items():
        curr_activations = stats.binned_statistic_2d(
            x=pos[:, 0],
            y=pos[:, 1],
            values=v.T,
            statistic="mean",
            bins=[arena_x_bins, arena_y_bins],
        )[0]
        model_activations[k] = np.transpose(curr_activations, (1, 2, 0))

    if model_pred_layer is not None:
        return model_activations[model_pred_layer]
    else:
        return model_activations


def configure_options(
    save_dir=None,
    rnn_type="UGRNN",
    activation="relu",
    place_cell_identity=False,
    cue_2d_input_kwargs=None,
    reward_zone_size=None,
    reward_zone_prob=1.0,
    # 15 cm from boundaries as in Butler et al. 2019
    reward_zone_x_offset=0.15,
    reward_zone_y_offset=0.15,
    reward_zone_min_x=None,
    reward_zone_max_x=None,
    reward_zone_min_y=None,
    reward_zone_max_y=None,
    reward_zone_navigate_timesteps=None,
    n_epochs=100,
    n_steps=1000,
    box_width=2.2,
    box_height=2.2,
    min_x=None,
    min_y=None,
    max_x=None,
    max_y=None,
    run_ID=None,
):
    class Options:
        pass

    options = Options()
    options.save_dir = save_dir

    if cue_2d_input_kwargs is not None:
        options.cue_2d_input_kwargs = cue_2d_input_kwargs

    if reward_zone_size is not None:
        options.reward_zone_size = reward_zone_size
        assert reward_zone_prob is not None
        options.reward_zone_prob = reward_zone_prob
        assert reward_zone_x_offset is not None
        options.reward_zone_x_offset = reward_zone_x_offset
        assert reward_zone_y_offset is not None
        options.reward_zone_y_offset = reward_zone_y_offset
        options.reward_zone_min_x = reward_zone_min_x
        options.reward_zone_max_x = reward_zone_max_x
        options.reward_zone_min_y = reward_zone_min_y
        options.reward_zone_max_y = reward_zone_max_y
        options.reward_zone_navigate_timesteps = reward_zone_navigate_timesteps
    options.place_cell_identity = (
        place_cell_identity  # decode positions directly if true
    )
    options.n_epochs = n_epochs  # number of training epochs
    options.n_steps = n_steps  # batches per epoch
    options.batch_size = 200  # number of trajectories per batch
    options.sequence_length = 20  # number of steps in trajectory
    options.learning_rate = 1e-4  # gradient descent learning rate
    if options.place_cell_identity:  # Cueva and Wei Position Loss
        options.Np = 2
    else:
        options.Np = 512  # number of place cells
    options.Ng = 4096  # number of grid cells
    options.place_cell_rf = 0.12  # width of place cell center tuning curve (m)
    options.surround_scale = 2  # if DoG, ratio of sigma2^2 to sigma1^2
    options.RNN_type = rnn_type
    options.activation = activation  # recurrent nonlinearity
    options.weight_decay = 1e-4  # strength of weight decay on recurrent weights
    options.DoG = True  # use difference of gaussians tuning curves
    options.box_width = box_width  # width of training environment (meters)
    options.box_height = box_height  # height of training environment (meters)
    if (
        (min_x is not None)
        and (max_x is not None)
        and (min_y is not None)
        and (max_y is not None)
    ):
        assert np.isclose((max_x - min_x), options.box_width)
        assert np.isclose((max_y - min_y), options.box_height)
        options.min_x = min_x
        options.max_x = max_x
        options.min_y = min_y
        options.max_y = max_y
    set_env_dims(options)
    options.run_ID = run_ID
    return options


def configure_model(options, rnn_type="rnn"):
    place_cells = PlaceCells(options)
    if rnn_type.lower() == "ugrnn":
        model = UGRNN(options=options, place_cells=place_cells)
    elif rnn_type.lower() == "cueugrnn":
        model = CueUGRNN(options=options, place_cells=place_cells)
    else:
        raise ValueError
    return model


def load_trained_model(
    load_dir=None,
    run_ID=None,
    ckpt_file=None,
    options=None,
    rnn_type="UGRNN",
    activation="relu",
    random_init=False,
    place_cell_identity=False,
    cue_2d_input_kwargs=None,
    **reward_zone_kwargs,
):

    if options is None:
        options = configure_options(
            save_dir=load_dir,
            rnn_type=rnn_type,
            activation=activation,
            place_cell_identity=place_cell_identity,
            run_ID=run_ID,
            cue_2d_input_kwargs=cue_2d_input_kwargs,
            **reward_zone_kwargs,
        )
    print(f"Configured these options {vars(options)}")

    trained_model = configure_model(options, rnn_type=rnn_type)
    if not random_init:
        ckpt_dir = os.path.join(options.save_dir, options.run_ID, "ckpts")
        assert os.path.isdir(ckpt_dir) is True
        ckpt = tf.train.Checkpoint(
            step=tf.Variable(0),
            optimizer=tf.keras.optimizers.Adam(learning_rate=options.learning_rate),
            net=trained_model,
        )
        ckpt_manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=500)
        if ckpt_file is None:
            latest_ckpt_path = ckpt_manager.latest_checkpoint
        else:
            latest_ckpt_path = os.path.join(ckpt_dir, ckpt_file)
        print(f"Loading ckpt from {latest_ckpt_path}")
        ckpt.restore(latest_ckpt_path).assert_existing_objects_matched()
    return trained_model
