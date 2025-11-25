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
import pytest
import numpy as np
from mpi4py import MPI
from nek5000reader import Nek5000Reader

@pytest.fixture
def data_dir():
    return os.path.join(os.path.dirname(__file__), 'data')

def test_initialization(data_dir):
    nek_file = os.path.join(data_dir, 'turbPipe.nek5000')
    reader = Nek5000Reader(nek_file)
    
    assert reader.num_timesteps == 10
    assert reader.first_timestep == 0
    assert reader.rank == 0  # Assuming serial run for tests
    
    info = reader.get_info()
    assert info['nek5000_file'] == nek_file
    assert info['num_blocks'] > 0

def test_read_timestep(data_dir):
    nek_file = os.path.join(data_dir, 'turbPipe.nek5000')
    reader = Nek5000Reader(nek_file)
    
    # Read timestep 0
    data = reader.read_timestep(0)
    
    assert 'coordinates' in data
    assert 'fields' in data
    assert 'time' in data
    assert 'cycle' in data
    
    # Check coordinates
    coords = data['coordinates']
    assert isinstance(coords, np.ndarray)
    assert coords.ndim == 2
    assert coords.shape[1] == 3
    assert coords.shape[0] > 0
    
    # Check fields
    fields = data['fields']
    assert isinstance(fields, dict)
    assert len(fields) > 0
    
    # Check metadata
    meta = data['metadata']
    assert meta['step'] == 0
    assert meta['has_mesh'] is True
