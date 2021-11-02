import numpy as np

ALPHA_RNG = list(
    np.geomspace(1e-9, 1e9, num=99, endpoint=True)
)  # note: this includes 1.0
L1_RATIO_RNG = [1e-16] + list(np.linspace(0.01, 1.0, num=99, endpoint=True))

ALPHA_RNG_SHORT = list(
    np.geomspace(1e-9, 1e9, num=9, endpoint=True)
)  # note: this includes 1.0
L1_RATIO_RNG_SHORT = [1e-16] + list(np.linspace(0.01, 1.0, num=9, endpoint=True))

# model kwargs used in the paper
## cues
CUE_1 = {"center_x": 0.0, "center_y": 0.0, "width": 0.3, "height": 0.3}
CUE_2 = {"center_x": 0.35, "center_y": 0.35, "width": 0.2, "height": 0.2}
CUE_3 = {"center_x": -0.3, "center_y": -0.3, "width": 0.1, "height": 0.1}
CUE_4 = {"center_x": 0.6, "center_y": 0.5, "width": 0.1, "height": 0.1}
CUE_5 = {"center_x": -0.7, "center_y": -0.6, "width": 0.06, "height": 0.06}
CUE_2D_INPUT_KWARGS = {
    "cue_extents": [CUE_1, CUE_2, CUE_3, CUE_4, CUE_5],
    "cue_prob": 0.5,
}

## rewards
REWARD_KWARGS = {
    "reward_zone_size": 0.2,
    "reward_zone_prob": 0.625,
    "reward_zone_min_x": 0.65,
    "reward_zone_max_x": 0.85,
    "reward_zone_min_y": 0.65,
    "reward_zone_max_y": 0.85,
    "reward_zone_navigate_timesteps": 7,
}
