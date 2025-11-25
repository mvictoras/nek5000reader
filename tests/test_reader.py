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
from nek5000reader import (
    Nek5000Reader,
    parse_nek5000_control,
    read_basic_header_and_endian,
    read_time_and_tags,
    parse_var_tags,
    read_block_ids,
    build_connectivity,
    build_step_filename,
    partition_blocks,
)
from nek5000reader.utils import last_int_in_string, read_ascii_token, peek


if MPI.COMM_WORLD.Get_size() > 1:
    pytest.skip("Non-MPI tests must be run with mpiexec -n 1", allow_module_level=True)

@pytest.fixture
def data_dir():
    return os.path.join(os.path.dirname(__file__), 'data')


@pytest.fixture
def stsabl_file(data_dir):
    return os.path.join(data_dir, 'stsabl.nek5000')


@pytest.fixture
def case_file(data_dir):
    return os.path.join(data_dir, 'stsabl.nek5000')


# ============================================================================
# CORE FUNCTIONALITY TESTS
# ============================================================================

def test_initialization(stsabl_file):
    """Test basic reader initialization"""
    reader = Nek5000Reader(stsabl_file)
    
    assert reader.num_timesteps == 1
    assert reader.first_timestep == 0
    assert reader.rank == 0  # Assuming serial run for tests
    
    info = reader.get_info()
    assert info['nek5000_file'] == stsabl_file
    assert info['num_blocks'] > 0


def test_read_timestep(stsabl_file):
    """Test reading a single timestep"""
    reader = Nek5000Reader(stsabl_file)
    
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
    assert 'metadata' in data
    meta = data['metadata']
    assert 'tags' in meta
    assert 'step' in meta
    assert 'has_mesh' in meta
    assert 'mesh_is_3d' in meta
    assert 'block_dims' in meta
    assert 'num_blocks_local' in meta
    assert 'num_blocks_global' in meta


    assert meta['step'] == 0
    assert meta['num_blocks_local'] > 0
    assert meta['num_blocks_global'] > 0


def test_read_timestep_without_mesh(stsabl_file):
    """Test reading timestep without mesh coordinates"""
    reader = Nek5000Reader(stsabl_file)
    data = reader.read_timestep(0, read_mesh=False)
    
    # Should not have coordinates or connectivity when read_mesh=False
    assert 'coordinates' not in data or data['coordinates'] is None
    assert 'connectivity' not in data or data['connectivity'] is None
    assert 'fields' in data
    assert len(data['fields']) > 0


def test_get_timestep_list_all(stsabl_file):
    """Test getting all timesteps"""
    reader = Nek5000Reader(stsabl_file)
    timesteps = reader.get_timestep_list()
    
    assert isinstance(timesteps, list)
    assert len(timesteps) == reader.num_timesteps
    assert timesteps[0] == reader.first_timestep


def test_get_timestep_list_with_range(stsabl_file):
    """Test getting timestep list with range specification"""
    reader = Nek5000Reader(stsabl_file)
    
    # Test with explicit range (even though we only have 1 timestep)
    timesteps = reader.get_timestep_list("0:1:1")
    assert isinstance(timesteps, list)
    assert 0 in timesteps


def test_get_info(stsabl_file):
    """Test get_info returns complete information"""
    reader = Nek5000Reader(stsabl_file)
    info = reader.get_info()
    
    # Check required fields
    assert 'nek5000_file' in info
    assert 'num_blocks' in info
    assert 'num_timesteps' in info
    assert 'first_timestep' in info
    assert 'block_dims' in info
    assert 'precision' in info
    assert 'mesh_is_3d' in info
    assert 'total_block_size' in info
    assert 'swap_endian' in info
    assert 'mpi_rank' in info
    assert 'mpi_size' in info
    assert 'blocks_this_rank' in info
    
    # Validate values
    assert info['num_blocks'] > 0
    assert info['num_timesteps'] > 0
    assert info['precision'] in [4, 8]
    assert info['total_block_size'] > 0
    assert info['mpi_rank'] >= 0
    assert info['mpi_size'] > 0
    assert info['blocks_this_rank'] > 0


def test_case_file_reading(case_file):
    """Test reading the case.nek5000 file"""
    reader = Nek5000Reader(case_file)
    
    assert reader.num_timesteps == 1
    assert reader.first_timestep == 0
    
    data = reader.read_timestep(0)
    assert 'coordinates' in data
    assert 'fields' in data
    assert data['coordinates'].shape[0] > 0


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

def test_invalid_file_path():
    """Test initialization with non-existent file"""
    with pytest.raises((FileNotFoundError, IOError, RuntimeError)):
        Nek5000Reader('/nonexistent/path/to/file.nek5000')


def test_invalid_timestep(stsabl_file):
    """Test reading invalid timestep number"""
    reader = Nek5000Reader(stsabl_file)
    
    # Try to read timestep beyond available range
    with pytest.raises((ValueError, IndexError, FileNotFoundError, RuntimeError)):
        reader.read_timestep(999)


def test_negative_timestep(stsabl_file):
    """Test reading negative timestep"""
    reader = Nek5000Reader(stsabl_file)
    
    with pytest.raises((ValueError, IndexError, FileNotFoundError, RuntimeError)):
        reader.read_timestep(-1)


# ============================================================================
# FIELD VALIDATION TESTS
# ============================================================================

def test_field_data_types(stsabl_file):
    """Test that field arrays have correct data types"""
    reader = Nek5000Reader(stsabl_file)
    data = reader.read_timestep(0)
    
    fields = data['fields']
    for field_name, field_data in fields.items():
        assert isinstance(field_data, np.ndarray)
        # Should be float32 as per implementation
        assert field_data.dtype in [np.float32, np.float64]


def test_velocity_field_structure(stsabl_file):
    """Test velocity field has correct structure"""
    reader = Nek5000Reader(stsabl_file)
    data = reader.read_timestep(0)
    
    fields = data['fields']
    if 'Velocity' in fields:
        vel = fields['Velocity']
        # Velocity should be flat array with 3*num_vertices elements
        num_verts = data['coordinates'].shape[0]
        assert vel.shape[0] == num_verts * 3


def test_velocity_magnitude_calculation(stsabl_file):
    """Test velocity magnitude is calculated correctly"""
    reader = Nek5000Reader(stsabl_file)
    data = reader.read_timestep(0)
    
    fields = data['fields']
    if 'Velocity' in fields and 'Velocity Magnitude' in fields:
        vel = fields['Velocity']
        vel_mag = fields['Velocity Magnitude']
        
        num_verts = data['coordinates'].shape[0]
        vx = vel[0*num_verts:1*num_verts]
        vy = vel[1*num_verts:2*num_verts]
        vz = vel[2*num_verts:3*num_verts]
        
        # Manually calculate magnitude
        expected_mag = np.sqrt(vx**2 + vy**2 + vz**2)
        
        # Should match within floating point tolerance
        np.testing.assert_allclose(vel_mag, expected_mag, rtol=1e-5)


def test_coordinate_array_shape(stsabl_file):
    """Test coordinate array has correct shape"""
    reader = Nek5000Reader(stsabl_file)
    data = reader.read_timestep(0)
    
    coords = data['coordinates']
    assert coords.ndim == 2
    assert coords.shape[1] == 3  # Always 3D (x, y, z)
    assert coords.dtype in [np.float32, np.float64]


# ============================================================================
# GEOMETRY TESTS
# ============================================================================

def test_connectivity_array(stsabl_file):
    """Test connectivity array structure"""
    reader = Nek5000Reader(stsabl_file)
    data = reader.read_timestep(0)
    
    if 'connectivity' in data:
        conn = data['connectivity']
        assert isinstance(conn, np.ndarray)
        assert conn.dtype == np.int32
        assert conn.ndim == 1
        
        # For 3D hex mesh: 8 vertices per cell
        # For 2D quad mesh: 4 vertices per cell
        info = reader.get_info()
        nx, ny, nz = info['block_dims']
        
        if nz > 1:  # 3D
            cells_per_block = (nx - 1) * (ny - 1) * (nz - 1)
            verts_per_cell = 8
        else:  # 2D
            cells_per_block = (nx - 1) * (ny - 1)
            verts_per_cell = 4
        
        expected_conn_size = info['num_blocks'] * cells_per_block * verts_per_cell
        assert conn.shape[0] == expected_conn_size


def test_build_connectivity_3d():
    """Test 3D hex connectivity generation"""
    nx, ny, nz = 3, 3, 3
    num_blocks = 2
    total_block_size = nx * ny * nz
    
    conn = build_connectivity((nx, ny, nz), num_blocks, total_block_size, mesh_is_3d=True)
    
    # Expected: (nx-1)*(ny-1)*(nz-1) cells per block * 8 vertices per hex * num_blocks
    cells_per_block = (nx - 1) * (ny - 1) * (nz - 1)
    expected_size = num_blocks * cells_per_block * 8
    
    assert conn.shape[0] == expected_size
    assert conn.dtype == np.int64
    assert np.all(conn >= 0)


def test_build_connectivity_2d():
    """Test 2D quad connectivity generation"""
    nx, ny, nz = 4, 4, 1
    num_blocks = 3
    total_block_size = nx * ny
    
    conn = build_connectivity((nx, ny, nz), num_blocks, total_block_size, mesh_is_3d=False)
    
    # Expected: (nx-1)*(ny-1) cells per block * 4 vertices per quad * num_blocks
    cells_per_block = (nx - 1) * (ny - 1)
    expected_size = num_blocks * cells_per_block * 4
    
    assert conn.shape[0] == expected_size
    assert conn.dtype == np.int64
    assert np.all(conn >= 0)


# ============================================================================
# HEADER PARSING TESTS
# ============================================================================

def test_parse_nek5000_control(stsabl_file):
    """Test parsing .nek5000 control file"""
    info = parse_nek5000_control(stsabl_file)
    
    assert 'filetemplate' in info
    assert 'firsttimestep' in info
    assert 'numtimesteps' in info
    
    assert isinstance(info['filetemplate'], str)
    assert isinstance(info['firsttimestep'], int)
    assert isinstance(info['numtimesteps'], int)
    assert info['numtimesteps'] > 0


def test_read_basic_header_and_endian(data_dir):
    """Test reading basic header and endianness detection"""
    # Read the actual data file
    data_file = os.path.join(data_dir, 'stsabl0.f00000')
    
    if os.path.exists(data_file):
        precision, block_dims, num_blocks, swap_endian = read_basic_header_and_endian(data_file)
        
        assert precision in [4, 8]
        assert isinstance(block_dims, tuple)
        assert len(block_dims) == 3
        assert all(d > 0 for d in block_dims)
        assert num_blocks > 0
        assert isinstance(swap_endian, bool)


def test_read_time_and_tags(data_dir):
    """Test reading time and variable tags"""
    data_file = os.path.join(data_dir, 'stsabl0.f00000')
    
    if os.path.exists(data_file):
        time, cycle, tags, has_mesh = read_time_and_tags(data_file)
        
        assert isinstance(time, (int, float))
        assert isinstance(cycle, int)
        assert isinstance(tags, str)
        assert isinstance(has_mesh, bool)


def test_parse_var_tags():
    """Test parsing variable tags"""
    # Test with typical Nek5000 tags
    tags = "X U P T"
    var_names, var_lens = parse_var_tags(tags, mesh_is_3d=True)
    
    assert isinstance(var_names, list)
    assert isinstance(var_lens, list)
    assert len(var_names) == len(var_lens)
    
    # Should have Velocity, Velocity Magnitude, Pressure, Temperature
    assert 'Velocity' in var_names or 'U' in tags
    
    # Test 2D case
    var_names_2d, var_lens_2d = parse_var_tags(tags, mesh_is_3d=False)
    assert isinstance(var_names_2d, list)


def test_read_block_ids(data_dir):
    """Test reading block IDs from file"""
    data_file = os.path.join(data_dir, 'stsabl0.f00000')
    
    if os.path.exists(data_file):
        precision, block_dims, num_blocks, swap_endian = read_basic_header_and_endian(data_file)
        block_ids = read_block_ids(data_file, num_blocks, swap_endian)
        
        assert isinstance(block_ids, np.ndarray)
        assert block_ids.shape[0] == num_blocks
        assert block_ids.dtype == np.int32


# ============================================================================
# UTILITY FUNCTION TESTS
# ============================================================================

def test_build_step_filename_single_specifier():
    """Test filename building with single printf specifier"""
    fmt = "data%05d.fld"
    result = build_step_filename(fmt, 42)
    assert result == "data00042.fld"


def test_build_step_filename_double_specifier():
    """Test filename building with two printf specifiers"""
    fmt = "turbPipe%01d.f%05d"
    result = build_step_filename(fmt, 123, dir_index=0)
    assert result == "turbPipe0.f00123"
    
    result = build_step_filename(fmt, 456, dir_index=1)
    assert result == "turbPipe1.f00456"


def test_build_step_filename_no_specifier():
    """Test filename building with no printf specifiers"""
    fmt = "datafile"
    result = build_step_filename(fmt, 10)
    assert result == "datafile10"


def test_last_int_in_string():
    """Test extracting last integer from string"""
    assert last_int_in_string("file123.txt") == 123
    assert last_int_in_string("data_v2_step456") == 456
    assert last_int_in_string("no_numbers_here") is None
    assert last_int_in_string("multiple12numbers34here56") == 56
    assert last_int_in_string("") is None


def test_read_ascii_token():
    """Test ASCII token reading from binary file"""
    import tempfile
    
    with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
        f.write(b"  token1  token2\ntoken3")
        temp_path = f.name
    
    try:
        with open(temp_path, 'rb') as f:
            token1 = read_ascii_token(f)
            assert token1 == "token1"
            
            token2 = read_ascii_token(f)
            assert token2 == "token2"
            
            token3 = read_ascii_token(f)
            assert token3 == "token3"
    finally:
        os.unlink(temp_path)


def test_peek_function():
    """Test peek function for binary files"""
    import tempfile
    
    with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
        f.write(b"ABC")
        temp_path = f.name
    
    try:
        with open(temp_path, 'rb') as f:
            # Peek should not advance position
            byte1 = peek(f)
            assert byte1 == ord('A')
            
            # Position should still be at start
            byte2 = peek(f)
            assert byte2 == ord('A')
            
            # Now actually read
            actual = f.read(1)
            assert actual == b'A'
            
            # Peek next byte
            byte3 = peek(f)
            assert byte3 == ord('B')
    finally:
        os.unlink(temp_path)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

def test_full_read_workflow(stsabl_file):
    """Test complete read workflow"""
    reader = Nek5000Reader(stsabl_file)
    
    # Get info
    info = reader.get_info()
    assert info['num_timesteps'] > 0
    
    # Get timestep list
    timesteps = reader.get_timestep_list()
    assert len(timesteps) > 0
    
    # Read first timestep
    data = reader.read_timestep(timesteps[0])
    
    # Verify all expected data is present
    assert 'coordinates' in data
    assert 'connectivity' in data
    assert 'fields' in data
    assert 'time' in data
    assert 'cycle' in data
    assert 'metadata' in data


def test_multiple_readers_same_file(stsabl_file):
    """Test multiple reader instances on same file"""
    reader1 = Nek5000Reader(stsabl_file)
    reader2 = Nek5000Reader(stsabl_file)
    
    data1 = reader1.read_timestep(0)
    data2 = reader2.read_timestep(0)
    
    # Both should read the same data
    np.testing.assert_array_equal(data1['coordinates'], data2['coordinates'])
    
    for field_name in data1['fields']:
        if field_name in data2['fields']:
            np.testing.assert_array_equal(
                data1['fields'][field_name],
                data2['fields'][field_name]
            )
