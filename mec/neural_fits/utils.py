import numpy as np
import os, copy
from mec.core.utils import all_disjoint, get_shape_2d


def unit_concat(d, arena_size, inner_key=None):
    import xarray as xr

    """For a given arena size, concatenates scalar values (e.g. neural predictivity of a model)
    for units across animals."""
    if inner_key is None:
        # e.g. grid scores
        return xr.concat([d[arena_size][a] for a in d[arena_size].keys()], dim="units")
    else:
        # e.g. neural predictivity of a model, so inner_key=metric, like "corr"
        return xr.concat(
            [
                d[arena_size][a][inner_key]
                for a in d[arena_size].keys()
                if inner_key in d[arena_size][a].keys()
            ],
            dim="units",
        )


def generate_train_test_splits(
    num_states, split_start_seed=0, num_splits=10, train_frac=0.2, shape_2d=None
):
    if isinstance(train_frac, str):
        # these are structured splits
        num_rows, num_cols = get_shape_2d(num_states=num_states, shape_2d=shape_2d)

        idx_mat = np.arange(num_states).reshape((num_rows, num_cols))
        train_test_splits = []
        if train_frac == "topbottom":
            half_1 = idx_mat[: (num_rows // 2), :].flatten()
            half_2 = idx_mat[(num_rows // 2) :, :].flatten()
        elif train_frac == "leftright":
            half_1 = idx_mat[:, : (num_cols // 2)].flatten()
            half_2 = idx_mat[:, (num_cols // 2) :].flatten()
        elif train_frac == "diag1":
            half_1 = idx_mat[np.triu_indices(n=num_rows, k=0, m=num_cols)].flatten()
            half_2 = idx_mat[np.tril_indices(n=num_rows, k=-1, m=num_cols)].flatten()
        elif train_frac == "diag2":
            idx_mat_2 = np.fliplr(idx_mat)
            half_1 = idx_mat_2[np.triu_indices(n=num_rows, k=0, m=num_cols)].flatten()
            half_2 = idx_mat_2[np.tril_indices(n=num_rows, k=-1, m=num_cols)].flatten()
        elif train_frac == "quad":
            quad1_test = idx_mat[: (num_rows // 2), : (num_cols // 2)].flatten()
            quad1_train = np.array(
                [x for x in idx_mat.flatten() if x not in quad1_test]
            )
            # check that it is a complete set
            assert set(list(quad1_train) + list(quad1_test)) == set(
                np.arange(num_states)
            )
            quad2_test = idx_mat[: (num_rows // 2), (num_cols // 2) :].flatten()
            quad2_train = np.array(
                [x for x in idx_mat.flatten() if x not in quad2_test]
            )
            # check that it is a complete set
            assert set(list(quad2_train) + list(quad2_test)) == set(
                np.arange(num_states)
            )
            quad3_test = idx_mat[(num_rows // 2) :, : (num_cols // 2)].flatten()
            quad3_train = np.array(
                [x for x in idx_mat.flatten() if x not in quad3_test]
            )
            # check that it is a complete set
            assert set(list(quad3_train) + list(quad3_test)) == set(
                np.arange(num_states)
            )
            quad4_test = idx_mat[(num_rows // 2) :, (num_cols // 2) :].flatten()
            quad4_train = np.array(
                [x for x in idx_mat.flatten() if x not in quad4_test]
            )
            # check that it is a complete set
            assert set(list(quad4_train) + list(quad4_test)) == set(
                np.arange(num_states)
            )
            # check that all quadrants are pairwise distinct
            assert (
                all_disjoint([quad1_test, quad2_test, quad3_test, quad4_test]) is True
            )
            train_test_splits.append({"train": quad1_train, "test": quad1_test})
            train_test_splits.append({"train": quad2_train, "test": quad2_test})
            train_test_splits.append({"train": quad3_train, "test": quad3_test})
            train_test_splits.append({"train": quad4_train, "test": quad4_test})
        else:
            raise ValueError

        if train_frac != "quad":
            # check that it is a complete set
            assert set(list(half_1) + list(half_2)) == set(np.arange(num_states))
            train_test_splits.append({"train": half_1, "test": half_2})
            train_test_splits.append({"train": half_2, "test": half_1})

    elif train_frac > 0:
        assert train_frac <= 1
        train_test_splits = []
        for s in range(num_splits):
            rand_idx = np.random.RandomState(seed=(split_start_seed + s)).permutation(
                num_states
            )
            num_train = (int)(np.ceil(train_frac * len(rand_idx)))
            train_idx = rand_idx[:num_train]
            test_idx = rand_idx[num_train:]
            curr_sp = {"train": train_idx, "test": test_idx}
            train_test_splits.append(curr_sp)
    else:
        print("Train fraction is 0, make sure your map has no parameters!")
        # we apply no random permutation in this case as there is no training of parameters (e.g. rsa)
        train_test_splits = [
            {"train": np.array([], dtype=int), "test": np.arange(num_states)}
        ]
    return train_test_splits


def nan_filter(source_resp, target_resp):
    """Helper function that first filters stimuli across all cells within a given animal
    so that it is all non-Nan. Then, filters the non-NaN stimuli across both source and target response."""
    # stimuli across all cells in a given animal/model that are non-Nan
    source_include = np.isfinite(source_resp).all(axis=-1)
    target_include = np.isfinite(target_resp).all(axis=-1)
    # compare across stimuli where both are non-NaN
    both_include = np.logical_and(source_include, target_include)
    source_resp = source_resp[both_include]
    target_resp = target_resp[both_include]
    return source_resp, target_resp


def prep_data_2d(X, Y):
    """Helper function that ensures data is non-NaN and flattened to be (num_stimuli, num_units)."""
    X = X.reshape((-1, X.shape[-1]))
    Y = Y.reshape((-1, Y.shape[-1]))
    X, Y = nan_filter(X, Y)
    assert X.shape[0] == Y.shape[0]
    return X, Y


def return_mean_sp_scores(map_type, map_kwargs, train_test_sp, X, Y, shape_2d=None):
    from mec.neural_mappers.pipeline_neural_map import PipelineNeuralMap

    """Returns the scores per train/test split for each metric type."""
    mean_scores = {"corr": []}
    if isinstance(map_kwargs, list):
        # a different kwarg per train test split
        assert len(map_kwargs) == len(train_test_sp)
    for curr_sp_idx, curr_sp in enumerate(train_test_sp):
        train_idx = curr_sp["train"]
        test_idx = curr_sp["test"]

        curr_map_kwargs = (
            map_kwargs[curr_sp_idx]
            if isinstance(map_kwargs, list)
            else copy.deepcopy(map_kwargs)
        )
        curr_map = PipelineNeuralMap(map_type=map_type, map_kwargs=curr_map_kwargs)
        curr_map.fit(X=X[train_idx], Y=Y[train_idx])
        Y_pred = curr_map.predict(X[test_idx])
        curr_sp_score = curr_map.score(Y=Y[test_idx], Y_pred=Y_pred)
        mean_scores["corr"].append(curr_sp_score)

    # average across train/test splits for each metric
    for metric_type in mean_scores.keys():
        mean_scores[metric_type] = np.nanmean(
            np.stack(mean_scores[metric_type], axis=0), axis=0
        )

    return mean_scores


def package_scores(scores, cell_ids):
    """If we have scores per unit, then we will associate
    each score with a unit as a single xarray."""
    if len(scores.shape) == 1:
        import xarray as xr

        scores = xr.DataArray(scores, coords={"units": cell_ids}, dims=["units"])
    return scores
