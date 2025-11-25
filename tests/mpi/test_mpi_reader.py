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

import os

import numpy as np
import pytest
from mpi4py import MPI

from nek5000reader import (Nek5000Reader, build_connectivity,
                           build_step_filename, parse_nek5000_control,
                           parse_var_tags, partition_blocks,
                           read_basic_header_and_endian, read_block_ids,
                           read_time_and_tags)
from nek5000reader.utils import last_int_in_string, peek, read_ascii_token

if MPI.COMM_WORLD.Get_size() == 1:
    pytest.skip("MPI tests must be run with mpiexec -n N", allow_module_level=True)


@pytest.fixture
def data_dir():
    return os.path.join(os.path.dirname(__file__), "../data")


@pytest.fixture
def stsabl_file(data_dir):
    return os.path.join(data_dir, "stsabl.nek5000")


@pytest.fixture
def case_file(data_dir):
    return os.path.join(data_dir, "stsabl.nek5000")


# ============================================================================
# MPI PARALLEL TESTS
# ============================================================================


def test_mpi_initialization(stsabl_file):
    """Test MPI-aware initialization"""
    comm = MPI.COMM_WORLD
    reader = Nek5000Reader(stsabl_file, comm=comm)

    assert reader.rank == comm.Get_rank()
    assert reader.size == comm.Get_size()


def test_mpi_block_distribution(stsabl_file):
    """Test that blocks are distributed across MPI ranks"""
    comm = MPI.COMM_WORLD
    reader = Nek5000Reader(stsabl_file, comm=comm)

    # Each rank should have some blocks (or none if more ranks than blocks)
    info = reader.get_info()
    total_blocks = info["num_blocks"]

    # Sum up blocks across all ranks
    local_blocks = info["blocks_this_rank"] if "blocks_this_rank" in info else 0
    all_blocks = comm.allreduce(local_blocks, op=MPI.SUM)

    # Total should equal original
    assert all_blocks == total_blocks


def test_mpi_parallel_read(stsabl_file):
    """Test parallel reading with MPI"""
    comm = MPI.COMM_WORLD
    reader = Nek5000Reader(stsabl_file, comm=comm)

    data = reader.read_timestep(0)

    # Each rank should have valid data
    assert "coordinates" in data
    assert "fields" in data

    # Verify data is not empty (unless rank has no blocks)
    if data["coordinates"] is not None:
        assert data["coordinates"].shape[0] > 0


def test_partition_blocks_even():
    """Test block partitioning with even distribution"""
    comm = MPI.COMM_WORLD
    num_blocks = 100

    counts, displs = partition_blocks(num_blocks, comm)

    assert isinstance(counts, np.ndarray)
    assert isinstance(displs, np.ndarray)
    assert counts.dtype == np.int32
    assert displs.dtype == np.int32
    assert len(counts) == comm.Get_size()
    assert len(displs) == comm.Get_size()
    assert np.sum(counts) == num_blocks


def test_partition_blocks_uneven():
    """Test block partitioning with uneven distribution"""
    comm = MPI.COMM_WORLD
    num_blocks = 10

    counts, displs = partition_blocks(num_blocks, comm)

    assert np.sum(counts) == num_blocks
    assert displs[0] == 0

    # Verify displacements are cumulative
    if comm.Get_size() > 1:
        for i in range(1, len(displs)):
            assert displs[i] == displs[i - 1] + counts[i - 1]
