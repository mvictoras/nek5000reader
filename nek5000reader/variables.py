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
Functions for reading variable data from Nek5000 files.
"""

from typing import Dict, List
import numpy as np

from .header import total_header_size_bytes


def read_variables_for_my_blocks(dfname: str,
                                 var_names: List[str],
                                 var_lens: List[int],
                                 my_block_positions: np.ndarray,
                                 totalBlockSize: int,
                                 mesh_is_3d: bool,
                                 precision: int,
                                 swapEndian: bool,
                                 has_mesh: bool,
                                 numBlocks_global: int) -> Dict[str, np.ndarray]:
    """
    Faster: use a single np.memmap per variable plane and gather all my blocks at once.
    - Endianness handled by dtype ('<' or '>').
    - 2D Velocity reads Vx,Vy from file and fills Vz=0.
    - Always returns float32 arrays.
    
    Args:
        dfname: Path to Nek5000 data file
        var_names: List of variable names
        var_lens: List of component counts for each variable
        my_block_positions: Array of block positions to read
        totalBlockSize: Size of each block
        mesh_is_3d: Whether mesh is 3D
        precision: Precision in bytes (4 or 8)
        swapEndian: Whether to swap byte order
        has_mesh: Whether mesh data is present
        numBlocks_global: Total number of blocks in file
        
    Returns:
        Dictionary mapping variable names to numpy arrays
    """
    result: Dict[str, np.ndarray] = {}

    comps_vel_in_file = 3 if mesh_is_3d else 2             # what the file actually stores for U
    comps_xyz = 3 if mesh_is_3d else 2
    header_bytes = total_header_size_bytes(numBlocks_global, totalBlockSize,
                                           comps_xyz, precision, has_mesh)

    # One scalar "plane" across *all* blocks in bytes
    plane_bytes = numBlocks_global * totalBlockSize * precision

    # Choose file dtype with explicit endianness
    endian = ">" if swapEndian else "<"
    dt = np.dtype(endian + ("f4" if precision == 4 else "f8"))

    nblk_local = int(my_block_positions.size)
    nverts = nblk_local * totalBlockSize

    # We compute offsets by counting how many "planes" we've passed in the file.
    # Velocity contributes `comps_vel_in_file` planes; each scalar (P, T, S##) contributes 1 plane.
    planes_before = 0

    need_vel_mag = ("Velocity Magnitude" in var_names)
    have_velocity = False
    vel_flat = None  # will hold [Vx...][Vy...][Vz...]

    i = 0
    while i < len(var_names):
        name = var_names[i]
        ncomp = var_lens[i]

        if name == "Velocity Magnitude":
            # We'll compute it after we read Velocity
            i += 1
            continue

        if name == "Velocity":
            # Map the contiguous velocity region (all blocks) as one 2D array
            offset = header_bytes + planes_before * plane_bytes
            # shape = (numBlocks_global, totalBlockSize * comps_vel_in_file)
            mm = np.memmap(dfname, dtype=dt, mode="r",
                           offset=offset,
                           shape=(numBlocks_global, totalBlockSize * comps_vel_in_file))

            sel = mm[my_block_positions]  # (nblk_local, totalBlockSize*comps_vel_in_file)

            # Build a flat 3-comp array [Vx...][Vy...][Vz...], always 3 comps in output
            vel_flat = np.empty(nverts * 3, dtype=np.float32)
            # X
            vel_flat[0*nverts:1*nverts] = sel[:, 0:totalBlockSize].reshape(-1).astype(np.float32, copy=False)
            # Y
            vel_flat[1*nverts:2*nverts] = sel[:, totalBlockSize:2*totalBlockSize].reshape(-1).astype(np.float32, copy=False)
            # Z
            if mesh_is_3d:
                vel_flat[2*nverts:3*nverts] = sel[:, 2*totalBlockSize:3*totalBlockSize].reshape(-1).astype(np.float32, copy=False)
            else:
                vel_flat[2*nverts:3*nverts] = 0.0

            result["Velocity"] = vel_flat  # already flat
            have_velocity = True

            planes_before += comps_vel_in_file
            i += 1
            continue

        # Scalar variable (P, T, S##, etc.)
        offset = header_bytes + planes_before * plane_bytes
        mm = np.memmap(dfname, dtype=dt, mode="r",
                       offset=offset,
                       shape=(numBlocks_global, totalBlockSize))

        sel = mm[my_block_positions]  # (nblk_local, totalBlockSize)
        result[name] = sel.reshape(-1).astype(np.float32, copy=False)

        planes_before += 1
        i += 1

    # Compute Velocity Magnitude if requested
    if need_vel_mag:
        if not have_velocity:
            raise RuntimeError("Requested 'Velocity Magnitude' but 'Velocity' was not present.")
        vx = vel_flat[0*nverts:1*nverts]
        vy = vel_flat[1*nverts:2*nverts]
        vz = vel_flat[2*nverts:3*nverts]
        result["Velocity Magnitude"] = np.sqrt(vx*vx + vy*vy + vz*vz, dtype=np.float32)

    return result
