# -*- coding: utf-8 -*-
import tensorflow as tf
import copy
import numpy as np
import itertools


class TrajectoryGenerator(object):
    def __init__(self, options, place_cells, trajectory_seed=None):
        self.options = options
        self.place_cells = place_cells
        self.trajectory_seed = trajectory_seed

        if hasattr(self.options, "cue_2d_input_kwargs"):
            self.cue_2d_input_kwargs = self.options.cue_2d_input_kwargs
        else:
            self.cue_2d_input_kwargs = None

        if hasattr(self.options, "reward_zone_size"):
            self.reward_zone_size = self.options.reward_zone_size
            self.reward_zone_prob = self.options.reward_zone_prob
            self.reward_zone_x_offset = self.options.reward_zone_x_offset
            self.reward_zone_y_offset = self.options.reward_zone_y_offset
            self.reward_zone_min_x = self.options.reward_zone_min_x
            self.reward_zone_max_x = self.options.reward_zone_max_x
            self.reward_zone_min_y = self.options.reward_zone_min_y
            self.reward_zone_max_y = self.options.reward_zone_max_y
            self.reward_zone_navigate_timesteps = (
                self.options.reward_zone_navigate_timesteps
            )
        else:
            self.reward_zone_size = None

    def avoid_wall(self, position, hd, min_x, max_x, min_y, max_y):
        """
        Compute distance and angle to nearest wall
        """
        x = position[:, 0]
        y = position[:, 1]
        dists = [max_x - x, max_y - y, x - min_x, y - min_y]
        d_wall = np.min(dists, axis=0)
        angles = np.arange(4) * np.pi / 2
        theta = angles[np.argmin(dists, axis=0)]
        hd = np.mod(hd, 2 * np.pi)
        a_wall = hd - theta
        a_wall = np.mod(a_wall + np.pi, 2 * np.pi) - np.pi

        is_near_wall = (d_wall < self.border_region) * (np.abs(a_wall) < np.pi / 2)
        turn_angle = np.zeros_like(hd)
        turn_angle[is_near_wall] = np.sign(a_wall[is_near_wall]) * (
            np.pi / 2 - np.abs(a_wall[is_near_wall])
        )

        return is_near_wall, turn_angle

    def _trajectory_builder(
        self,
        samples,
        batch_size,
        min_x,
        max_x,
        min_y,
        max_y,
        dt=0.02,
        start_x=None,
        start_y=None,
        position_pred_start_idx=2,
    ):

        """Build a random walk in a rectangular box, with given inputs"""

        sigma = 5.76 * 2  # stdev rotation velocity (rads/sec)
        b = 0.13 * 2 * np.pi  # forward velocity rayleigh dist scale (m/sec)
        mu = 0  # turn angle bias
        self.border_region = 0.03  # meters

        # Initialize variables
        position = np.zeros([batch_size, samples + 2, 2])
        head_dir = np.zeros([batch_size, samples + 2])
        if start_x is None:
            position[:, 0, 0] = np.random.uniform(
                low=min_x, high=max_x, size=batch_size
            )
        else:
            assert len(start_x.shape) == 1
            assert start_x.shape[0] == batch_size
            position[:, 0, 0] = start_x

        # at t = 0, we start in a random position and head direction, for each episode
        # we have batch_size total episodes
        if start_y is None:
            position[:, 0, 1] = np.random.uniform(
                low=min_y, high=max_y, size=batch_size
            )
        else:
            assert len(start_y.shape) == 1
            assert start_y.shape[0] == batch_size
            position[:, 0, 1] = start_y

        head_dir[:, 0] = np.random.uniform(low=0, high=2 * np.pi, size=batch_size)
        # Generate sequence of random boosts and turns
        random_turn = np.random.normal(mu, sigma, [batch_size, samples + 1])

        # at t = 0, we start at 0 velocity
        velocity = np.zeros([batch_size, samples + 2])

        random_vel = np.random.rayleigh(b, [batch_size, samples + 1])
        v = np.abs(np.random.normal(0, b * np.pi / 2, batch_size))

        for t in range(samples + 1):
            # Update velocity
            v = random_vel[:, t]
            turn_angle = np.zeros(batch_size)

            # If in border region, turn and slow down
            is_near_wall, turn_angle = self.avoid_wall(
                position=position[:, t],
                hd=head_dir[:, t],
                min_x=min_x,
                max_x=max_x,
                min_y=min_y,
                max_y=max_y,
            )
            v[is_near_wall] *= 0.25

            # Update turn angle
            turn_angle += dt * random_turn[:, t]

            # Take a step
            velocity[:, t] = v * dt
            update = velocity[:, t, None] * np.stack(
                [np.cos(head_dir[:, t]), np.sin(head_dir[:, t])], axis=-1
            )
            position[:, t + 1] = position[:, t] + update

            # Rotate head direction
            head_dir[:, t + 1] = head_dir[:, t] + turn_angle

        head_dir = np.mod(head_dir + np.pi, 2 * np.pi) - np.pi  # Periodic variable

        assert position_pred_start_idx >= 1

        traj = {}
        # Input variables
        # we get head direction at t = 0
        # since we compute angular velocity from it,
        # whose first element is hd at t = 1 - hd at t = 0
        # traj['init_hd'] = head_dir[:,0,None]
        # we get the first position after moving one step
        # with nonzero velocity (since at t=0 our velocity is 0)
        traj["init_x"] = position[:, position_pred_start_idx - 1, 0, None]
        traj["init_y"] = position[:, position_pred_start_idx - 1, 1, None]

        # get the first nonzero velocity
        # up to second to last velocity, which will be our input
        # since we predict position as our target is 1 timestep ahead of input
        traj["ego_v"] = velocity[:, position_pred_start_idx - 1 : -1]
        # ang_v = np.diff(head_dir, axis=-1)
        # traj['phi_x'], traj['phi_y'] = np.cos(ang_v)[:,:-1], np.sin(ang_v)[:,:-1]

        # Target variables
        traj["target_hd"] = head_dir[:, position_pred_start_idx - 1 : -1]
        traj["target_x"] = position[:, position_pred_start_idx:, 0]
        traj["target_y"] = position[:, position_pred_start_idx:, 1]

        return traj, position

    def get_cue_2d_extent(self, cue_data):
        cue_min_x = cue_data["center_x"] - (cue_data["width"] / 2.0)
        cue_max_x = cue_data["center_x"] + (cue_data["width"] / 2.0)
        cue_min_y = cue_data["center_y"] - (cue_data["height"] / 2.0)
        cue_max_y = cue_data["center_y"] + (cue_data["height"] / 2.0)
        return cue_min_x, cue_max_x, cue_min_y, cue_max_y

    def _check_cue_2d_extent(self, cue_data, min_x, max_x, min_y, max_y):
        cue_min_x, cue_max_x, cue_min_y, cue_max_y = self.get_cue_2d_extent(
            cue_data=cue_data
        )
        assert (cue_min_x >= min_x) and (cue_max_x <= max_x)
        assert (cue_min_y >= min_y) and (cue_max_y <= max_y)

    def get_cue_2d_distances(
        self, curr_pos_x, curr_pos_y, cue_extents, distance_mask=None
    ):
        """distance of 0 if in cue region, else distance to closest boundary
        returns a distance of shape (batch_size, num_cues).
        cue_extents is a list of (batch_size,) vectors (or single scalar if fixed across episodes)
        of the location of cue_1,..., cue_{len(cue_extents)}."""

        assert curr_pos_x.shape == curr_pos_y.shape
        assert len(curr_pos_x.shape) == 1  # should be of shape (batch_size,)

        cue_distances = []
        for curr_cue in cue_extents:
            d_x = np.maximum(
                np.abs(curr_pos_x - curr_cue["center_x"]) - (curr_cue["width"] / 2.0),
                0.0,
            )
            d_y = np.maximum(
                np.abs(curr_pos_y - curr_cue["center_y"]) - (curr_cue["height"] / 2.0),
                0.0,
            )
            cue_distance = np.sqrt(np.square(d_x) + np.square(d_y))
            cue_distances.append(cue_distance)
        # (batch_size, num_cues)
        cue_distances = np.stack(cue_distances, axis=-1)
        if distance_mask is not None:
            # zero out cues on a per episode (batch) basis
            assert distance_mask.shape == cue_distances.shape
            cue_distances *= distance_mask
        return cue_distances

    def get_cue_inputs_2d(self, traj, cue_extents, distance_mask=None):
        assert traj["target_x"].shape == traj["target_y"].shape
        assert len(traj["target_x"].shape) == 2

        batch_size = traj["target_x"].shape[0]
        samples = traj["target_x"].shape[1]
        # initialize cue inputs
        num_cues = len(cue_extents)
        cue_inputs = np.zeros([batch_size, samples, num_cues])

        cue_inputs[:, 0, :] = self.get_cue_2d_distances(
            curr_pos_x=traj["init_x"][:, 0],
            curr_pos_y=traj["init_y"][:, 0],
            cue_extents=cue_extents,
            distance_mask=distance_mask,
        )
        for t in range(1, samples):
            # line up with velocity timestep (current timestep, not the timestep ahead whose position we are trying to predict)
            cue_inputs[:, t, :] = self.get_cue_2d_distances(
                curr_pos_x=traj["target_x"][:, t - 1],
                curr_pos_y=traj["target_y"][:, t - 1],
                cue_extents=cue_extents,
                distance_mask=distance_mask,
            )

        return cue_inputs

    def check_cue_2d_disjoint(self, cue1, cue2):
        # returns True if the cues do not overlap and false if they do
        cue1_min_x, cue1_max_x, cue1_min_y, cue1_max_y = self.get_cue_2d_extent(
            cue_data=cue1
        )
        cue2_min_x, cue2_max_x, cue2_min_y, cue2_max_y = self.get_cue_2d_extent(
            cue_data=cue2
        )
        return (
            (cue1_max_x < cue2_min_x)
            or (cue1_min_x > cue2_max_x)
            or (cue1_max_y < cue2_min_y)
            or (cue1_min_y > cue2_max_y)
        )

    def generate_trajectory(self, batch_size, min_x, max_x, min_y, max_y):
        """Generate a random walk in a rectangular box"""
        if self.trajectory_seed is not None:
            np.random.seed(self.trajectory_seed)

        samples = self.options.sequence_length
        dt = 0.02  # time step increment (seconds)
        traj, position = self._trajectory_builder(
            samples=samples,
            batch_size=batch_size,
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
            dt=dt,
        )

        if self.cue_2d_input_kwargs is not None:
            # for now we do not support reward zones since we would need to support models with both inputs
            # and the code currently resets traj but would need to preserve cue inputs
            assert self.reward_zone_size is None

            cue_extents = self.cue_2d_input_kwargs["cue_extents"]
            assert isinstance(cue_extents, list)
            assert len(cue_extents) > 0
            # fixed set of cues to use per episode, check that each cue is within the environment
            for cue_data in cue_extents:
                self._check_cue_2d_extent(
                    cue_data, min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y
                )
            # check that the cues are disjoint
            cue_pairs = list(itertools.combinations(cue_extents, r=2))
            assert all(
                [
                    self.check_cue_2d_disjoint(cue1=cue_pair[0], cue2=cue_pair[1])
                    for cue_pair in cue_pairs
                ]
            )

            cue_prob = self.cue_2d_input_kwargs.get("cue_prob", 0.5)
            num_cues = len(cue_extents)
            distance_mask = np.zeros([batch_size, num_cues])
            with_cue_mask = np.ones([batch_size, num_cues])
            # pick a random proportion of episodes to have cues
            num_cue_episodes = (int)(np.ceil(cue_prob * batch_size))
            cue_episode_batch_idxs = np.random.permutation(batch_size)[
                :num_cue_episodes
            ]
            distance_mask[cue_episode_batch_idxs] = with_cue_mask[
                cue_episode_batch_idxs
            ]
            # set distances
            traj["cue_inputs"] = self.get_cue_inputs_2d(
                traj=traj, cue_extents=cue_extents, distance_mask=distance_mask
            )
        # rewards if any (2d environment only)
        elif self.reward_zone_size is not None:
            start_x = traj["init_x"][:, 0]
            start_y = traj["init_y"][:, 0]
            ego_v_x = traj["ego_v"] * np.cos(traj["target_hd"])
            ego_v_y = traj["ego_v"] * np.sin(traj["target_hd"])
            # 0 rewards by default
            rewards = np.zeros([batch_size, samples + 2], dtype=traj["ego_v"].dtype)
            assert self.reward_zone_prob is not None
            assert (self.reward_zone_prob > 0) and (self.reward_zone_prob <= 1)
            assert self.reward_zone_x_offset is not None
            assert self.reward_zone_y_offset is not None
            # get the allowed bounds of the reward zone
            # (sometimes we want the reward zone to be offset from the walls of the arena)
            reward_min_x = min_x + self.reward_zone_x_offset
            reward_max_x = max_x - self.reward_zone_x_offset
            reward_min_y = min_y + self.reward_zone_y_offset
            reward_max_y = max_y - self.reward_zone_y_offset

            # check that reward zone is inside box, taking into account the specified offsets
            assert self.reward_zone_min_x is not None
            assert self.reward_zone_max_x is not None
            assert self.reward_zone_min_y is not None
            assert self.reward_zone_max_y is not None
            assert self.reward_zone_min_x >= reward_min_x
            assert self.reward_zone_max_x <= reward_max_x
            assert self.reward_zone_min_y >= reward_min_y
            assert self.reward_zone_max_y <= reward_max_y
            # check that reward zone dimensions match prescribed size
            assert np.isclose(
                (self.reward_zone_max_x - self.reward_zone_min_x),
                self.reward_zone_size,
            )
            assert np.isclose(
                (self.reward_zone_max_y - self.reward_zone_min_y),
                self.reward_zone_size,
            )

            reward_zone_min_x = self.reward_zone_min_x * np.ones(batch_size)
            reward_zone_max_x = self.reward_zone_max_x * np.ones(batch_size)
            reward_zone_min_y = self.reward_zone_min_y * np.ones(batch_size)
            reward_zone_max_y = self.reward_zone_max_y * np.ones(batch_size)

            # pick a random proportion of episodes to have reward zones
            num_reward_zones = (int)(np.ceil(self.reward_zone_prob * batch_size))
            reward_zone_batch_idxs = np.random.permutation(batch_size)[
                :num_reward_zones
            ]

            assert self.reward_zone_navigate_timesteps is not None
            # we now compute the velocity and head direction needed to get to the reward zone from its start position
            # the goal is the reward zone center at each episode
            goal_x = reward_zone_min_x + ((reward_zone_max_x - reward_zone_min_x) / 2.0)
            goal_y = reward_zone_min_y + ((reward_zone_max_y - reward_zone_min_y) / 2.0)

            assert self.reward_zone_navigate_timesteps <= samples
            goal_velocity_x = np.divide(
                (goal_x - start_x), self.reward_zone_navigate_timesteps
            )
            goal_velocity_y = np.divide(
                (goal_y - start_y), self.reward_zone_navigate_timesteps
            )

            # tile in time (batch x time)
            # we pad by two to avoid indexing errors when adding to position, then discard later
            goal_velocity_x = np.tile(
                np.expand_dims(goal_velocity_x, axis=-1),
                reps=(1, self.reward_zone_navigate_timesteps + 2),
            )
            goal_velocity_y = np.tile(
                np.expand_dims(goal_velocity_y, axis=-1),
                reps=(1, self.reward_zone_navigate_timesteps + 2),
            )

            # random walk inside reward zone for rest of the time
            # position 0 here is goal_x, goal_y, so we start at 1
            reward_zone_pos_start_idx = 1
            (reward_zone_traj, reward_zone_position,) = self._trajectory_builder(
                samples=(samples + 1)
                - (
                    self.reward_zone_navigate_timesteps + 2
                ),  # position 0 of the original position is ignored, so we go up to samples+1 to get to samples+2 length
                batch_size=batch_size,
                min_x=reward_zone_min_x,
                max_x=reward_zone_max_x,
                min_y=reward_zone_min_y,
                max_y=reward_zone_max_y,
                start_x=goal_x,
                start_y=goal_y,
                position_pred_start_idx=reward_zone_pos_start_idx,
                dt=dt,
            )
            # fill in the positions up to getting to the goal location
            navigate_position = copy.deepcopy(position)
            for t in range(2, self.reward_zone_navigate_timesteps + 2):
                navigate_position[:, t, 0] = (
                    navigate_position[:, t - 1, 0] + goal_velocity_x[:, t - 1]
                )
                navigate_position[:, t, 1] = (
                    navigate_position[:, t - 1, 1] + goal_velocity_y[:, t - 1]
                )

            # sanity check we are close to the goal at the end, across all episodes
            assert np.isclose(
                navigate_position[:, self.reward_zone_navigate_timesteps + 1, 0],
                goal_x,
            ).all()
            assert np.isclose(
                navigate_position[:, self.reward_zone_navigate_timesteps + 1, 1],
                goal_y,
            ).all()

            # fill in the remaining timesteps with random walk within the reward zone
            navigate_position[
                :, self.reward_zone_navigate_timesteps + 2 :, :
            ] = reward_zone_position[:, reward_zone_pos_start_idx:, :]

            # update the positions only for the episodes that have reward zones, else a standard random walk across the entire arena
            position[reward_zone_batch_idxs] = navigate_position[reward_zone_batch_idxs]

            traj = {}
            traj["init_x"] = np.expand_dims(start_x, axis=-1)
            traj["init_y"] = np.expand_dims(start_y, axis=-1)
            # concatenate across time
            reward_zone_velocity_x = reward_zone_traj["ego_v"] * np.cos(
                reward_zone_traj["target_hd"]
            )
            reward_zone_velocity_y = reward_zone_traj["ego_v"] * np.sin(
                reward_zone_traj["target_hd"]
            )
            # we discard last velocity as it is unused
            navigate_ego_v_x = np.concatenate(
                [goal_velocity_x[:, 1:-1], reward_zone_velocity_x], axis=-1
            )
            navigate_ego_v_y = np.concatenate(
                [goal_velocity_y[:, 1:-1], reward_zone_velocity_y], axis=-1
            )
            # update the velocities only for the episodes that have reward zones
            ego_v_x[reward_zone_batch_idxs] = navigate_ego_v_x[reward_zone_batch_idxs]
            ego_v_y[reward_zone_batch_idxs] = navigate_ego_v_y[reward_zone_batch_idxs]
            traj["ego_v_x"] = ego_v_x
            traj["ego_v_y"] = ego_v_y

            # Target variables
            traj["target_x"] = position[:, 2:, 0]
            traj["target_y"] = position[:, 2:, 1]
            assert traj["target_x"].shape[1] == traj["ego_v_x"].shape[1]
            assert traj["target_y"].shape[1] == traj["ego_v_y"].shape[1]
            # final consistency check that velocities and positions are lined up
            curr_pos_x = traj["init_x"][:, 0]
            curr_pos_y = traj["init_y"][:, 0]
            num_seq_steps = traj["ego_v_x"].shape[1]
            for t in range(num_seq_steps):
                curr_pos_x = curr_pos_x + traj["ego_v_x"][:, t]
                curr_pos_y = curr_pos_y + traj["ego_v_y"][:, t]
                assert np.isclose(curr_pos_x, traj["target_x"][:, t]).all()
                assert np.isclose(curr_pos_y, traj["target_y"][:, t]).all()

            assert position.shape[1] == samples + 2
            for t in range(samples + 2):
                reward_x = np.logical_and(
                    (reward_zone_min_x <= position[:, t, 0]),
                    (position[:, t, 0] <= reward_zone_max_x),
                )
                reward_y = np.logical_and(
                    (reward_zone_min_y <= position[:, t, 1]),
                    (position[:, t, 1] <= reward_zone_max_y),
                )
                batch_rewards = np.logical_and(reward_x, reward_y)
                # 1 if in reward zone and 0 if not
                if "ego_v" not in traj.keys():
                    v_dtype = traj["ego_v_x"].dtype
                else:
                    v_dtype = traj["ego_v"].dtype
                batch_rewards = batch_rewards.astype(v_dtype)
                # update with the rewards only for the episodes that have reward zones
                rewards[reward_zone_batch_idxs, t] = batch_rewards[
                    reward_zone_batch_idxs
                ]

            # give rewards at same timestep as velocity
            traj["rewards"] = rewards[:, 1:-1]

        if self.trajectory_seed is not None:
            # increment it for next time
            self.trajectory_seed += 1
        return traj

    def get_batch(
        self,
        batch_size=None,
        box_width=None,
        box_height=None,
        min_x=None,
        max_x=None,
        min_y=None,
        max_y=None,
    ):
        """ returns a batch of sample trajectories"""
        if batch_size is None:
            batch_size = self.options.batch_size

        if (min_x is None) or (max_x is None):
            if box_width is not None:
                min_x = -box_width / 2.0
                max_x = box_width / 2.0
            else:
                min_x = self.options.min_x
                max_x = self.options.max_x

        if (min_y is None) or (max_y is None):
            if box_height is not None:
                min_y = -box_height / 2.0
                max_y = box_height / 2.0
            else:
                min_y = self.options.min_y
                max_y = self.options.max_y

        traj = self.generate_trajectory(
            batch_size=batch_size, min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y
        )

        if "ego_v" not in traj.keys():
            # this means we already computed the x and y egocentric velocity
            v = tf.stack([traj["ego_v_x"], traj["ego_v_y"]], axis=-1)
        else:
            v = tf.stack(
                [
                    traj["ego_v"] * tf.cos(traj["target_hd"]),
                    traj["ego_v"] * tf.sin(traj["target_hd"]),
                ],
                axis=-1,
            )

        pos = tf.stack([traj["target_x"], traj["target_y"]], axis=-1)
        place_outputs = self.place_cells.get_activation(pos)

        init_pos = tf.stack([traj["init_x"], traj["init_y"]], axis=-1)
        init_actv = tf.squeeze(self.place_cells.get_activation(init_pos))

        if self.cue_2d_input_kwargs is not None:
            inputs = (v, init_actv, tf.identity(traj["cue_inputs"]))
        else:
            inputs = (v, init_actv)

        return (inputs, pos, place_outputs)

    def get_generator(self, **kwargs):
        """
        Returns a generator that yields batches of trajectories
        """
        while True:
            (inputs, pos, place_outputs) = self.get_batch(**kwargs)

            yield (inputs, place_outputs, pos)
