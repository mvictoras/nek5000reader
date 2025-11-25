# Copyright (c) 2025 Victor Mateevitsi. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""
Main Nek5000Reader class for high-level file reading.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from mpi4py import MPI

from .header import (
    parse_nek5000_control,
    read_basic_header_and_endian,
    read_time_and_tags,
    parse_var_tags,
    read_block_ids,
    read_map_file,
)
from .geometry import read_coords_for_my_blocks, build_connectivity
from .variables import read_variables_for_my_blocks
from .utils import build_step_filename, partition_blocks


class Nek5000Reader:
    """
    High-level reader for Nek5000 simulation files.

    Supports both serial and MPI-parallel reading of Nek5000 data files.

    Example:
        >>> from mpi4py import MPI
        >>> reader = Nek5000Reader("simulation.nek5000", MPI.COMM_WORLD)
        >>> data = reader.read_timestep(0)
        >>> coords = data['coordinates']
        >>> velocity = data['fields']['Velocity']
    """

    def __init__(self, nek5000_file: str, comm: Optional[MPI.Comm] = None):
        """
        Initialize the Nek5000 reader.

        Args:
            nek5000_file: Path to .nek5000 control file
            comm: MPI communicator (default: MPI.COMM_WORLD)
        """
        self.nek5000_file = nek5000_file
        self.comm = comm if comm is not None else MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        # Parse control file
        self.ctrl = parse_nek5000_control(nek5000_file)
        self.filetemplate = self.ctrl["filetemplate"]
        self.first_timestep = self.ctrl["firsttimestep"]
        self.num_timesteps = self.ctrl["numtimesteps"]

        # Read header from first file
        first_file = build_step_filename(self.filetemplate, self.first_timestep)
        self.precision, self.block_dims, self.num_blocks, self.swap_endian = (
            read_basic_header_and_endian(first_file)
        )

        self.mesh_is_3d = self.block_dims[2] > 1
        self.total_block_size = (
            self.block_dims[0] * self.block_dims[1] * self.block_dims[2]
        )

        # Read block IDs and map file
        self.block_ids_all = read_block_ids(
            first_file, self.num_blocks, self.swap_endian
        )
        map_ids = read_map_file(nek5000_file)
        self.global_order = (
            map_ids
            if (map_ids is not None and map_ids.size == self.num_blocks)
            else self.block_ids_all
        )

        # Partition blocks across MPI ranks
        counts, displs = partition_blocks(self.num_blocks, self.comm)
        self.my_count = int(counts[self.rank])
        self.my_ids = self.global_order[
            displs[self.rank] : displs[self.rank] + self.my_count
        ].copy()

        # Build block position lookup
        self.my_positions = self._build_position_lookup()

        # Cache for geometry (read once, reuse)
        self._coords_cache = None
        self._conn_cache = None

    def _build_position_lookup(self) -> np.ndarray:
        """Build fast lookup from block IDs to positions."""
        ids = self.block_ids_all.astype(np.int64, copy=False)
        my_ids64 = self.my_ids.astype(np.int64, copy=False)
        max_id = int(ids.max())

        if max_id <= 2 * ids.size:
            # Use direct lookup table
            lut = np.full(max_id + 1, -1, dtype=np.int64)
            lut[ids] = np.arange(ids.size, dtype=np.int64)
            my_positions = lut[my_ids64]
            if np.any(my_positions < 0):
                raise RuntimeError("Some my_ids not found in block_ids_all")
        else:
            # Use binary search
            order = np.argsort(ids)
            sorted_ids = ids[order]
            loc = np.searchsorted(sorted_ids, my_ids64)
            if not np.all(sorted_ids[loc] == my_ids64):
                raise RuntimeError("Some my_ids not found in block_ids_all")
            my_positions = order[loc]

        return my_positions.astype(np.int64, copy=False)

    def get_timestep_list(self, step_range: Optional[str] = None) -> List[int]:
        """
        Get list of timesteps to process.

        Args:
            step_range: Range string like 'start:end:stride' (default: all timesteps)

        Returns:
            List of timestep numbers
        """
        if step_range:
            seg = [int(x) for x in step_range.split(":")]
            if len(seg) == 1:
                return [seg[0]]
            elif len(seg) == 2:
                return list(range(seg[0], seg[1]))
            else:
                return list(range(seg[0], seg[1], seg[2]))
        else:
            return list(
                range(self.first_timestep, self.first_timestep + self.num_timesteps)
            )

    def read_timestep(self, step: int, read_mesh: bool = True) -> Dict:
        """
        Read a single timestep.

        Args:
            step: Timestep number to read
            read_mesh: Whether to read mesh coordinates (default: True)

        Returns:
            Dictionary containing:
                - 'time': simulation time
                - 'cycle': cycle number
                - 'coordinates': (N, 3) array of vertex coordinates (if read_mesh=True)
                - 'connectivity': connectivity array (if read_mesh=True)
                - 'fields': dictionary of field arrays
                - 'metadata': additional metadata
        """
        df = build_step_filename(self.filetemplate, step)
        time_val, cycle_val, tags, has_mesh = read_time_and_tags(df)

        result = {
            "time": time_val,
            "cycle": cycle_val,
            "metadata": {
                "tags": tags,
                "has_mesh": has_mesh,
                "step": step,
                "mesh_is_3d": self.mesh_is_3d,
                "block_dims": self.block_dims,
                "num_blocks_local": self.my_count,
                "num_blocks_global": self.num_blocks,
            },
        }

        # Read geometry (cache if possible)
        if read_mesh:
            if self._coords_cache is None:
                mesh_src = (
                    df
                    if has_mesh
                    else build_step_filename(self.filetemplate, self.first_timestep)
                )
                coords = read_coords_for_my_blocks(
                    mesh_src,
                    self.my_positions,
                    self.total_block_size,
                    self.mesh_is_3d,
                    self.precision,
                    self.swap_endian,
                    self.num_blocks,
                )
                conn = build_connectivity(
                    self.block_dims,
                    self.my_count,
                    self.total_block_size,
                    self.mesh_is_3d,
                )
                # Optimize connectivity dtype
                if conn.dtype != np.int32 and int(conn.max()) < np.iinfo(np.int32).max:
                    conn = conn.astype(np.int32, copy=False)

                self._coords_cache = coords
                self._conn_cache = conn

            result["coordinates"] = self._coords_cache
            result["connectivity"] = self._conn_cache

        # Read variables
        var_names, var_lens = parse_var_tags(tags, self.mesh_is_3d)
        fields = read_variables_for_my_blocks(
            df,
            var_names,
            var_lens,
            self.my_positions,
            self.total_block_size,
            self.mesh_is_3d,
            self.precision,
            self.swap_endian,
            has_mesh,
            self.num_blocks,
        )
        result["fields"] = fields

        return result

    def get_info(self) -> Dict:
        """
        Get information about the dataset.

        Returns:
            Dictionary with dataset information
        """
        return {
            "nek5000_file": self.nek5000_file,
            "filetemplate": self.filetemplate,
            "first_timestep": self.first_timestep,
            "num_timesteps": self.num_timesteps,
            "precision": self.precision,
            "block_dims": self.block_dims,
            "num_blocks": self.num_blocks,
            "mesh_is_3d": self.mesh_is_3d,
            "total_block_size": self.total_block_size,
            "swap_endian": self.swap_endian,
            "mpi_rank": self.rank,
            "mpi_size": self.size,
            "blocks_this_rank": self.my_count,
        }
