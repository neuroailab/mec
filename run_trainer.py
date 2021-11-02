import tensorflow as tf
import os
from mec.models.utils import configure_options, configure_model
from mec.models.trainer import Trainer
from mec.core.constants import CUE_2D_INPUT_KWARGS

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu", type=str, default=None, required=True, help="What gpu to use."
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        required=True,
        help="Directory to save to.",
    )
    parser.add_argument(
        "--run_ID", type=str, default=None, required=True, help="Experiment ID."
    )
    parser.add_argument(
        "--rnn_type", type=str, default="UGRNN", choices=["UGRNN", "CueUGRNN"]
    )
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument(
        "--place_cell_identity",
        type=bool,
        default=False,
        help="Use the Banino, Barry et al. Place Cell loss by default. Otherwise, if True, use the Cueva & Wei Position Loss.",
    )
    parser.add_argument(
        "--train_with_cues",
        type=bool,
        default=False,
        help="Train path integrator with cue inputs along with velocity input.",
    )
    parser.add_argument(
        "--cue_prob",
        type=float,
        default=None,
        help="Fraction of episodes to have cues. If None, uses the default used in the paper (0.5).",
    )
    parser.add_argument(
        "--reward_zone_size",
        type=float,
        default=None,
        help="Specify a size (in meters) if you want to train the model with rewards.",
    )
    parser.add_argument(
        "--reward_zone_prob",
        type=float,
        default=0.625,
        help="Fraction of episodes to have rewards. The default value is the optimal epsilon in the paper, which is 1-reward_zone_prob.",
    )
    parser.add_argument(
        "--reward_zone_min_x",
        type=float,
        default=0.65,
        help="Where to place the reward zone (in x, in meters).",
    )
    parser.add_argument(
        "--reward_zone_max_x",
        type=float,
        default=0.85,
        help="Where to place the reward zone (in x, in meters).",
    )
    parser.add_argument(
        "--reward_zone_min_y",
        type=float,
        default=0.65,
        help="Where to place the reward zone (in y, in meters).",
    )
    parser.add_argument(
        "--reward_zone_max_y",
        type=float,
        default=0.85,
        help="Where to place the reward zone (in y, in meters).",
    )
    parser.add_argument(
        "--reward_zone_navigate_timesteps",
        type=int,
        default=7,
        help="Number of timesteps to get to center of reward zone before taking a random walk within it.",
    )
    ARGS = parser.parse_args()

    # If GPUs available, select which to train on
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ARGS.gpu

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    # Training options and hyperparameters
    cue_2d_input_kwargs = None
    if ARGS.train_with_cues:
        assert ARGS.rnn_type.lower() == "cueugrnn"
        # the cues used in the paper, feel free to use your own
        cue_2d_input_kwargs = CUE_2D_INPUT_KWARGS
        if ARGS.cue_prob is not None:
            cue_2d_input_kwargs["cue_prob"] = ARGS.cue_prob
    options = configure_options(
        rnn_type=ARGS.rnn_type,
        activation=ARGS.activation,
        place_cell_identity=ARGS.place_cell_identity,
        cue_2d_input_kwargs=cue_2d_input_kwargs,
        save_dir=ARGS.save_dir,
        run_ID=ARGS.run_ID,
        reward_zone_size=ARGS.reward_zone_size,
        reward_zone_prob=ARGS.reward_zone_prob,
        reward_zone_min_x=ARGS.reward_zone_min_x,
        reward_zone_max_x=ARGS.reward_zone_max_x,
        reward_zone_min_y=ARGS.reward_zone_min_y,
        reward_zone_max_y=ARGS.reward_zone_max_y,
        reward_zone_navigate_timesteps=ARGS.reward_zone_navigate_timesteps,
    )

    model = configure_model(options, rnn_type=ARGS.rnn_type)
    trainer = Trainer(options, model)
    trainer.train(n_epochs=options.n_epochs, n_steps=options.n_steps, save=True)
