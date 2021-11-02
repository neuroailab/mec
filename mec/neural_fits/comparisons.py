import numpy as np
import os, copy
import itertools
from mec.neural_fits.utils import (
    generate_train_test_splits,
    prep_data_2d,
    return_mean_sp_scores,
    package_scores,
)
from mec.models.utils import get_model_activations


def fit_subroutine(
    X,
    Y,
    map_type,
    map_kwargs,
    train_frac,
    num_train_test_splits,
    split_start_seed=0,
    shape_2d=None,
):

    X, Y = prep_data_2d(X=X, Y=Y)

    # we generate train/test splits for each source, target pair
    # since the stimuli can be different for each pair (when we don't smooth firing rates)
    train_test_sp = generate_train_test_splits(
        num_states=X.shape[0],
        train_frac=train_frac,
        num_splits=num_train_test_splits,
        shape_2d=shape_2d,
        split_start_seed=split_start_seed,
    )

    mean_scores = return_mean_sp_scores(
        map_type=map_type,
        map_kwargs=map_kwargs,
        train_test_sp=train_test_sp,
        X=X,
        Y=Y,
        shape_2d=shape_2d,
    )
    return mean_scores


def compare_animals(
    arena_sizes,
    dataset,
    map_type="sklinear",
    map_kwargs={},
    train_frac=0.2,
    num_train_test_splits=10,
    split_start_seed=0,
    mode="holdout",
    map_kwargs_per_cell=None,
    shape_2d=None,
):

    """Compare animals to each other with a given map"""
    source_dataset = copy.deepcopy(dataset)

    print("Comparing across animals")
    score_results = {}
    for arena_size in arena_sizes:

        score_results[arena_size] = {}
        arena_animals = list(dataset[arena_size].keys())
        if len(arena_animals) > 1:
            for animal in arena_animals:
                score_results[arena_size][animal] = {}
                if mode == "pairwise":
                    assert (
                        map_kwargs_per_cell is None
                    )  # currently unsupported since these kwargs were found via cross val in holdout mode
                    for animal_pair in itertools.permutations(arena_animals, r=2):
                        source_animal = animal_pair[0]
                        target_animal = animal_pair[1]
                        if target_animal == animal:
                            source_animal_resp = source_dataset[arena_size][
                                source_animal
                            ]["resp"]

                            target_animal_resp = dataset[arena_size][target_animal][
                                "resp"
                            ]

                            mean_scores = fit_subroutine(
                                X=source_animal_resp,
                                Y=target_animal_resp,
                                map_type=map_type,
                                map_kwargs=map_kwargs,
                                train_frac=train_frac,
                                num_train_test_splits=num_train_test_splits,
                                split_start_seed=split_start_seed,
                                shape_2d=shape_2d,
                            )

                            for metric_type in mean_scores.keys():
                                if (
                                    metric_type
                                    not in score_results[arena_size][animal].keys()
                                ):
                                    score_results[arena_size][animal][metric_type] = [
                                        mean_scores[metric_type]
                                    ]
                                else:
                                    score_results[arena_size][animal][
                                        metric_type
                                    ].append(mean_scores[metric_type])

                elif mode == "holdout":
                    """Concatenate neurons from source animal and fit to target."""
                    curr_source_animals = list(set(arena_animals) - set([animal]))
                    assert animal not in curr_source_animals
                    assert len(curr_source_animals) == len(arena_animals) - 1

                    holdout_source_animal_resp = np.concatenate(
                        [
                            source_dataset[arena_size][source_animal]["resp"]
                            for source_animal in curr_source_animals
                        ],
                        axis=-1,
                    )
                    target_animal_resp = dataset[arena_size][animal]["resp"]

                    subroutine_kwargs = {
                        "X": holdout_source_animal_resp,
                        "map_type": map_type,
                        "num_train_test_splits": num_train_test_splits,
                        "split_start_seed": split_start_seed,
                        "shape_2d": shape_2d,
                    }
                    if map_kwargs_per_cell is not None:
                        # no subselection allowed in this case, since these kwargs were found via cross validation on holdout animal between source and target
                        target_animal_cell_ids = dataset[arena_size][animal]["cell_ids"]
                        mean_scores = {}
                        for n, curr_target_cell_id in enumerate(target_animal_cell_ids):
                            curr_subroutine_kwargs = copy.deepcopy(subroutine_kwargs)
                            if len(target_animal_resp.shape) == 3:
                                curr_subroutine_kwargs["Y"] = np.expand_dims(
                                    target_animal_resp[:, :, n], axis=-1
                                )
                            elif len(target_animal_resp.shape) == 2:
                                curr_subroutine_kwargs["Y"] = np.expand_dims(
                                    target_animal_resp[:, n], axis=-1
                                )
                            else:
                                raise ValueError
                            curr_subroutine_kwargs["map_kwargs"] = map_kwargs_per_cell[
                                curr_target_cell_id
                            ]["map_kwargs"]
                            # can specify a train frac per cell, otherwise it globally uses the one passed in above
                            curr_subroutine_kwargs["train_frac"] = map_kwargs_per_cell[
                                curr_target_cell_id
                            ].get("train_frac", train_frac)
                            curr_mean_scores = fit_subroutine(**curr_subroutine_kwargs)
                            mean_scores[curr_target_cell_id] = curr_mean_scores
                    else:
                        subroutine_kwargs["Y"] = target_animal_resp
                        subroutine_kwargs["map_kwargs"] = map_kwargs
                        subroutine_kwargs["train_frac"] = train_frac
                        mean_scores = fit_subroutine(**subroutine_kwargs)
                        score_results[arena_size][animal] = mean_scores
                else:
                    raise ValueError

        for animal in score_results[arena_size].keys():
            metrics = (
                ["corr"]
                if map_kwargs_per_cell is not None
                else score_results[arena_size][animal].keys()
            )
            for metric_type in metrics:
                if mode == "pairwise":
                    # average across source animals
                    score_results[arena_size][animal][metric_type] = np.nanmean(
                        np.stack(
                            score_results[arena_size][animal][metric_type], axis=0
                        ),
                        axis=0,
                    )

                pkg_cell_ids = dataset[arena_size][animal]["cell_ids"]
                if map_kwargs_per_cell is not None:
                    mean_scores_arr = np.array(
                        [
                            np.squeeze(
                                score_results[arena_size][animal][curr_target_cell_id][
                                    metric_type
                                ]
                            )
                            for curr_target_cell_id in score_results[arena_size][
                                animal
                            ].keys()
                        ]
                    )
                    score_results[arena_size][animal][metric_type] = package_scores(
                        mean_scores_arr, cell_ids=pkg_cell_ids
                    )
                else:
                    score_results[arena_size][animal][metric_type] = package_scores(
                        score_results[arena_size][animal][metric_type],
                        cell_ids=pkg_cell_ids,
                    )

    return score_results


def compare_model(
    dataset,
    model=None,
    model_resp=None,
    cfg=None,
    arena_sizes=None,
    map_type="sklinear",
    map_kwargs={},
    train_frac=0.2,
    num_train_test_splits=10,
    split_start_seed=0,
    model_pred_layer="g",
    n_eval_seq=100,
    trajectory_seed=0,
    shape_2d=None,
    map_kwargs_per_cell=None,
    **bin_kwargs,
):

    """Compare model features to animal with a given map"""

    if model_resp is None:
        assert cfg is not None

    if not isinstance(cfg, list):
        cfg = [cfg]

    if not isinstance(trajectory_seed, list):
        trajectory_seed = [trajectory_seed]

    score_results = {}
    for arena_size in arena_sizes:
        print(f"Comparing to animals with arenas of size {arena_size} meters")

        if model_resp is None:
            model_resp = []
            for curr_seed in trajectory_seed:
                for curr_cfg in cfg:
                    assert curr_cfg.box_height == arena_size
                    assert curr_cfg.box_width == arena_size
                    model_act = get_model_activations(
                        model=model,
                        cfg=curr_cfg,
                        n_eval_seq=n_eval_seq,
                        model_pred_layer=model_pred_layer,
                        trajectory_seed=curr_seed,
                        **bin_kwargs,
                    )
                    num_units = model_act.shape[-1]
                    model_act = model_act.reshape((-1, num_units))
                    model_resp.append(model_act)
            model_resp = np.concatenate(model_resp, axis=0)
        else:
            print("Using passed in model response")
            # only one saved out model response for each arena size
            assert len(arena_sizes) == 1
            assert len(cfg) == 1
            assert len(trajectory_seed) == 1
            if len(model_resp.shape) == 3:
                num_units = model_resp.shape[-1]
                model_resp = model_resp.reshape((-1, num_units))
                print(f"Passed in model response reshaped to {model_resp.shape}")
            else:
                assert len(model_resp.shape) == 2

        score_results[arena_size] = {}
        arena_animals = list(dataset[arena_size].keys())
        for target_animal in arena_animals:
            target_animal_resp = dataset[arena_size][target_animal]["resp"]
            target_animal_cell_ids = dataset[arena_size][target_animal]["cell_ids"]
            subroutine_kwargs = {
                "X": model_resp,
                "map_type": map_type,
                "num_train_test_splits": num_train_test_splits,
                "split_start_seed": split_start_seed,
                "shape_2d": shape_2d,
            }

            if map_kwargs_per_cell is not None:
                mean_scores = {}
                for n, curr_target_cell_id in enumerate(target_animal_cell_ids):
                    curr_subroutine_kwargs = copy.deepcopy(subroutine_kwargs)
                    if len(target_animal_resp.shape) == 3:
                        curr_subroutine_kwargs["Y"] = np.expand_dims(
                            target_animal_resp[:, :, n], axis=-1
                        )
                    elif len(target_animal_resp.shape) == 2:
                        curr_subroutine_kwargs["Y"] = np.expand_dims(
                            target_animal_resp[:, n], axis=-1
                        )
                    else:
                        raise ValueError
                    curr_subroutine_kwargs["map_kwargs"] = map_kwargs_per_cell[
                        curr_target_cell_id
                    ]["map_kwargs"]
                    # can specify a train frac per cell, otherwise it globally uses the one passed in above
                    curr_subroutine_kwargs["train_frac"] = map_kwargs_per_cell[
                        curr_target_cell_id
                    ].get("train_frac", train_frac)
                    curr_mean_scores = fit_subroutine(**curr_subroutine_kwargs)
                    mean_scores[curr_target_cell_id] = curr_mean_scores
            else:
                subroutine_kwargs["Y"] = target_animal_resp
                subroutine_kwargs["map_kwargs"] = map_kwargs
                subroutine_kwargs["train_frac"] = train_frac
                mean_scores = fit_subroutine(**subroutine_kwargs)

            metrics = (
                ["corr"] if map_kwargs_per_cell is not None else mean_scores.keys()
            )
            for metric_type in metrics:
                if map_kwargs_per_cell is not None:
                    mean_scores_arr = np.array(
                        [
                            np.squeeze(mean_scores[curr_target_cell_id][metric_type])
                            for curr_target_cell_id in target_animal_cell_ids
                        ]
                    )
                    mean_scores[metric_type] = package_scores(
                        mean_scores_arr, cell_ids=target_animal_cell_ids
                    )
                else:
                    mean_scores[metric_type] = package_scores(
                        mean_scores[metric_type], cell_ids=target_animal_cell_ids
                    )

            score_results[arena_size][target_animal] = mean_scores

    return score_results


def get_fits(
    dataset,
    model=None,
    model_resp=None,
    cfg=None,
    arena_sizes=None,
    map_type="sklinear",
    map_kwargs={},
    train_frac=0.2,
    num_train_test_splits=10,
    split_start_seed=0,
    n_eval_seq=100,
    interanimal_mode="holdout",
    trajectory_seed=0,
    model_pred_layer="g",
    shape_2d=None,
    map_kwargs_per_cell=None,
    **bin_kwargs,
):

    if (model is None) and (model_resp is None):  # compare across animals
        score_results = compare_animals(
            arena_sizes=arena_sizes,
            dataset=dataset,
            map_type=map_type,
            map_kwargs=map_kwargs,
            train_frac=train_frac,
            num_train_test_splits=num_train_test_splits,
            split_start_seed=split_start_seed,
            mode=interanimal_mode,
            shape_2d=shape_2d,
            map_kwargs_per_cell=map_kwargs_per_cell,
        )
    else:
        score_results = compare_model(
            arena_sizes=arena_sizes,
            dataset=dataset,
            model=model,
            model_resp=model_resp,
            cfg=cfg,
            n_eval_seq=n_eval_seq,
            trajectory_seed=trajectory_seed,
            model_pred_layer=model_pred_layer,
            map_type=map_type,
            map_kwargs=map_kwargs,
            train_frac=train_frac,
            num_train_test_splits=num_train_test_splits,
            split_start_seed=split_start_seed,
            shape_2d=shape_2d,
            map_kwargs_per_cell=map_kwargs_per_cell,
            **bin_kwargs,
        )

    return score_results
