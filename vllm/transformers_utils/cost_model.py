# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import pickle
import glob
from typing import Optional

import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp1d

class ProfileComponent:
    ATTN_PROJ = "attn_proj"
    ATTN = "attention"
    GATE = "gate"
    EXPERT = "expert"

class ModelComponent:
    ATTENTION_GATE = "attn_gate"
    EXPERT = "expert"

# hardcode profile file names for now
_PROF_COMP_TO_FILE_NAME = {
    ProfileComponent.ATTN_PROJ: "mixtral_attention_projection",
    ProfileComponent.ATTN: "mixtral_paged_attention",
    ProfileComponent.GATE: "mixtral_gate",
    ProfileComponent.EXPERT: "expert_computation",
}


class ProfileBasedCostModel(object):
    """
    Cost model for a single LM layer's computation time (decode) based
    on profiled data. Profiles are generated through microbenchmarks.
    """

    def __init__(
        self,
        profile_dir=None,
        _timing_data=None,
    ) -> None:
        self.profile_dir = profile_dir
        # metadata
        self._metadata = {}
        if _timing_data is None:
            # read profile data from file
            self.timing_data = {}
            self._read_profile_data()
        else:
            self.timing_data = _timing_data
        # create interpolators
        self.exec_time_interpolators = {}

        for stage_key, data in self.timing_data.items():
            if stage_key == ProfileComponent.ATTN:
                # 2d interpolation
                interp = self._create_interpolator_2d(data)
            else:
                # 1d interpolation
                interp = self._create_interpolator_1d(data)
            self.exec_time_interpolators[stage_key] = interp

    def _create_interpolator_2d(self, data):
        # organize data into grid of [bs, context_len, data]
        batch_sizes = set()
        context_lens = set()
        values_min = float("inf")
        values_max = float("-inf")
        for bs, cl, v in data:
            batch_sizes.add(bs)
            context_lens.add(cl)
            values_min = min(values_min, v)
            values_max = max(values_max, v)
        # we extend the mbs and context_lens by one element to disallow
        # extrapolation beyond maximum values profiled
        batch_sizes = sorted(batch_sizes)
        batch_sizes = batch_sizes + [batch_sizes[-1] * 2]
        context_lens = sorted(context_lens)
        context_lens = context_lens + [context_lens[-1] * 2]
        # create interpolators
        x_shape = (len(batch_sizes), len(context_lens))
        # don't use infinity as fill value here, as during interpolation
        # it may get multiplied by a zero weight, resulting in nan
        # use a large number instead
        value_array = np.full(x_shape, fill_value=1e100)
        for bs, cl, v in data:
            x_coords = (batch_sizes.index(bs), context_lens.index(cl))
            value_array[x_coords] = v
        # for each batch size, if value for a context length is not available,
        # use the result from 1d interpolation
        def _find_next_valid_value(array, i):
            for j in range(i+1, len(array)):
                if array[j] < 1e99:
                    return array[j], j
            return None, None
        for bs_idx, bs in enumerate(batch_sizes[:-1]):
            for i in range(len(context_lens) - 1):
                if value_array[bs_idx, i] >= 1e99:
                    if i == 0:
                        # use the first reasonable value
                        next_valid_value, _ = _find_next_valid_value(value_array[bs_idx], i)
                        if next_valid_value is None:
                            raise ValueError(
                                f"Cannot find valid value for bs={bs}"
                            )
                        value_array[bs_idx, i] = next_valid_value
                    else:
                        # interpolate
                        next_valid_value, next_value_idx = _find_next_valid_value(value_array[bs_idx], i)
                        if next_valid_value is None:
                            # use the last reasonable value
                            value_array[bs_idx, i] = value_array[bs_idx, i-1]
                        else:
                            # linear interpolate
                            value_array[bs_idx, i] =  (
                                value_array[bs_idx, i-1] +
                                (value_array[bs_idx, next_value_idx] - value_array[bs_idx, i-1]) *
                                (context_lens[i] - context_lens[i-1]) /
                                (context_lens[next_value_idx] - context_lens[i-1])
                            )
        # allow extrapolation
        return RegularGridInterpolator(
                (batch_sizes, context_lens),
                value_array,
                method="linear",
                bounds_error=False,
                fill_value=None,
            )

    def _create_interpolator_1d(self, data):
        # organize data into grid of [bs, data]
        batch_sizes = set()
        values_min = float("inf")
        values_max = float("-inf")
        for bs, v in data:
            batch_sizes.add(bs)
            values_min = min(values_min, v)
            values_max = max(values_max, v)
        # we extend the mbs by one element to disallow
        # extrapolation beyond maximum values profiled
        batch_sizes = sorted(batch_sizes)
        batch_sizes = batch_sizes + [batch_sizes[-1] * 2]
        # create interpolators
        x_shape = (len(batch_sizes),)
        # don't use infinity as fill value here, as during interpolation
        # it may get multiplied by a zero weight, resulting in nan
        # use a large number instead
        value_array = np.full(x_shape, fill_value=1e100)
        for bs, v in data:
            x_coords = (batch_sizes.index(bs),)
            value_array[x_coords] = v
        # allow extrapolation
        return interp1d(
            batch_sizes,
            value_array,
            kind="linear",
            bounds_error=False,
            fill_value=None,
        )

    def _read_profile_data(self):
        """Read profile data from file."""
        if self.profile_dir is None:
            raise ValueError("Profile directory must be provided "
                             "for reading profile data.")
        profile_paths = glob.glob(os.path.join(self.profile_dir, "*.csv"))
        print(f"Found {profile_paths}")
        assert len(profile_paths) == 4, (
            f"Expected 4 profile files, got {len(profile_paths)}"
        )
        for profile_path in profile_paths:
            if _PROF_COMP_TO_FILE_NAME[ProfileComponent.ATTN] in profile_path:
                data = []
                with open(profile_path, 'r') as f:
                    # skip the first line
                    f.readline()
                    # read the rest lines
                    for line in f:
                        bs, context_len, lat = line.strip().split(",")
                        bs = int(bs)
                        context_len = int(context_len) # total number of tokens
                        lat = float(lat)
                        data.append((bs, context_len, lat))
                self.timing_data[ProfileComponent.ATTN] = data
            elif _PROF_COMP_TO_FILE_NAME[ProfileComponent.ATTN_PROJ] in profile_path:
                data = []
                with open(profile_path, 'r') as f:
                    # skip the first line
                    f.readline()
                    # read the rest lines
                    for line in f:
                        _, bs, lat = line.strip().split(",")
                        bs = int(bs)
                        lat = float(lat)
                        data.append((bs, lat))
                self.timing_data[ProfileComponent.ATTN_PROJ] = data
            elif _PROF_COMP_TO_FILE_NAME[ProfileComponent.GATE] in profile_path:
                data = []
                with open(profile_path, 'r') as f:
                    # skip the first line
                    f.readline()
                    # read the rest lines
                    for line in f:
                        bs, lat = line.strip().split(",")
                        bs = int(bs)
                        lat = float(lat)
                        data.append((bs, lat))
                self.timing_data[ProfileComponent.GATE] = data
            elif _PROF_COMP_TO_FILE_NAME[ProfileComponent.EXPERT] in profile_path:
                data = []
                with open(profile_path, 'r') as f:
                    # skip the first line
                    f.readline()
                    # read the rest lines
                    for line in f:
                        _, _, bs, lat, _ = line.strip().split(",")
                        bs = int(bs)
                        lat = float(lat)
                        data.append((bs, lat))
                self.timing_data[ProfileComponent.EXPERT] = data
            else:
                raise ValueError(f"Unknown profile file {profile_path}")

    def get_cost(
        self,
        stage: ModelComponent,
        batch_size: int,
        context_len: Optional[int] = None,
    ):
        """Get the computation cost of the stage in milliseconds (ms),
        under given micro-batch size and/or context length.
        """
        if stage == ModelComponent.ATTENTION_GATE:
            if context_len is None:
                raise ValueError(
                    "Context length must be provided for attention gate."
                )
            t = 0
            # attn
            attn_xcoords = (batch_size, context_len)
            t += float(self.exec_time_interpolators[ProfileComponent.ATTN](attn_xcoords))
            # attn proj
            # t += float(self.exec_time_interpolators[ProfileComponent.ATTN_PROJ](batch_size))
            # gate
            # t += float(self.exec_time_interpolators[ProfileComponent.GATE](batch_size))
            return t
        elif stage == ModelComponent.EXPERT:
            t = 0
            # expert
            t += float(self.exec_time_interpolators[ProfileComponent.EXPERT](batch_size))
            return t
        else:
            raise ValueError(
                f"Stage {stage} is not supported."
            )

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(dict(self.timing_data), f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            timing_data = pickle.load(f)
        return cls(_timing_data=timing_data)