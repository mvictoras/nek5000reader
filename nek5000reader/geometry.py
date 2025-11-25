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
Functions for reading and building mesh geometry.
"""

from typing import Tuple

import numpy as np


def read_coords_for_my_blocks(
    dfname: str,
    my_block_positions: np.ndarray,
    totalBlockSize: int,
    mesh_is_3d: bool,
    precision: int,
    swapEndian: bool,
    numBlocks_global: int,
) -> np.ndarray:
    """
    Fast path: map the entire coords region once, then gather in one shot.

    Args:
        dfname: Path to Nek5000 data file
        my_block_positions: Array of block positions to read
        totalBlockSize: Size of each block
        mesh_is_3d: Whether mesh is 3D
        precision: Precision in bytes (4 or 8)
        swapEndian: Whether to swap byte order
        numBlocks_global: Total number of blocks in file

    Returns:
        Array of coordinates with shape (num_vertices, 3)
    """
    comps = 3 if mesh_is_3d else 2
    nblk_local = int(my_block_positions.size)
    # Output (host, native float32)
    out = np.empty((nblk_local * totalBlockSize, 3), dtype=np.float32)

    # Byte offsets & element counts for the coordinates region
    header_bytes = 136 + numBlocks_global * 4  # 136-byte header + block-id table
    stride_elems = totalBlockSize * comps

    # Select dtype with the file's endianness
    # swapEndian == True  -> file is big endian
    # swapEndian == False -> file is little endian
    endian = ">" if swapEndian else "<"
    dt = np.dtype(endian + ("f4" if precision == 4 else "f8"))

    # Map the full coordinates region as a 2D array: (numBlocks_global, stride_elems)
    mm = np.memmap(
        dfname,
        dtype=dt,
        mode="r",
        offset=header_bytes,
        shape=(numBlocks_global, stride_elems),
    )

    # Vectorized gather of all my blocks (creates a dense ndarray)
    sel = mm[my_block_positions]  # shape: (nblk_local, stride_elems)

    # De-interleave into X/Y/Z in a single pass
    # Reshape output for clean slicing [block, point, coord]
    out3 = out.reshape(nblk_local, totalBlockSize, 3)
    out3[:, :, 0] = sel[:, 0:totalBlockSize]  # X
    out3[:, :, 1] = sel[:, totalBlockSize : 2 * totalBlockSize]  # Y
    if mesh_is_3d:
        out3[:, :, 2] = sel[:, 2 * totalBlockSize : 3 * totalBlockSize]  # Z
    else:
        out3[:, :, 2].fill(0.0)

    # If the file was double precision or non-native endian,
    # the assignment above casts to native float32 for you.
    return out


def build_connectivity(
    blockDims: Tuple[int, int, int],
    myNumBlocks: int,
    totalBlockSize: int,
    mesh_is_3d: bool,
) -> np.ndarray:
    """
    Vectorized connectivity builder.
    - Builds one block's connectivity with NumPy.
    - Broadcast-adds a per-block vertex offset (totalBlockSize) and tiles across myNumBlocks.

    Args:
        blockDims: Tuple of (nx, ny, nz) dimensions
        myNumBlocks: Number of blocks for this rank
        totalBlockSize: Size of each block
        mesh_is_3d: Whether mesh is 3D

    Returns:
        1D int64 array suitable for Conduit Blueprint 'connectivity'
    """
    nx, ny, nz = blockDims
    pts_per_block = np.int64(totalBlockSize)

    if mesh_is_3d:
        # Cells per block: (nx-1)*(ny-1)*(nz-1)
        i = np.arange(nx - 1, dtype=np.int64)
        j = np.arange(ny - 1, dtype=np.int64)
        k = np.arange(nz - 1, dtype=np.int64)
        # Order matches original loops: ii (slow), jj, kk (fast)
        I, J, K = np.meshgrid(i, j, k, indexing="ij")  # shapes: (nx-1, ny-1, nz-1)
        base = (K * (ny * nx) + J * nx + I).reshape(-1)

        layer = np.int64(nx * ny)
        # Hex corner pattern:
        # [p, p+1, p+nx+1, p+nx,  p+layer, p+layer+1, p+layer+nx+1, p+layer+nx]
        conn_block = np.stack(
            [
                base,
                base + 1,
                base + nx + 1,
                base + nx,
                base + layer,
                base + layer + 1,
                base + layer + nx + 1,
                base + layer + nx,
            ],
            axis=1,
        )  # (cells_per_block, 8)

        # Tile to all my blocks with vertex offsets
        offsets = (np.arange(myNumBlocks, dtype=np.int64) * pts_per_block)[
            :, None, None
        ]
        conn = conn_block[None, :, :] + offsets  # (myNumBlocks, cells_per_block, 8)
        return conn.reshape(-1)

    else:
        # 2D quads per block: (nx-1)*(ny-1)
        i = np.arange(nx - 1, dtype=np.int64)
        j = np.arange(ny - 1, dtype=np.int64)
        # Order matches original loops: ii (slow), jj (fast)
        I, J = np.meshgrid(i, j, indexing="ij")  # shapes: (nx-1, ny-1)
        base = (J * nx + I).reshape(-1)

        # Quad corner pattern:
        # [p, p+1, p+nx+1, p+nx]
        conn_block = np.stack(
            [base, base + 1, base + nx + 1, base + nx], axis=1
        )  # (cells_per_block, 4)

        offsets = (np.arange(myNumBlocks, dtype=np.int64) * pts_per_block)[
            :, None, None
        ]
        conn = conn_block[None, :, :] + offsets  # (myNumBlocks, cells_per_block, 4)
        return conn.reshape(-1)
