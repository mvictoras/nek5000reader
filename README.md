# Nek5000 Reader

[![Tests](https://github.com/mvictoras/nek5000reader/actions/workflows/tests.yml/badge.svg)](https://github.com/mvictoras/nek5000reader/actions/workflows/tests.yml)
[![PyPI version](https://badge.fury.io/py/nek5000reader.svg)](https://badge.fury.io/py/nek5000reader)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![codecov](https://codecov.io/gh/mvictoras/nek5000reader/branch/main/graph/badge.svg)](https://codecov.io/gh/mvictoras/nek5000reader)

A high-performance Python library for reading and processing Nek5000 simulation files.
Supports both serial and MPI-parallel reading with optimized memory-mapped I/O.

## Features

- 🚀 **Fast I/O**: Memory-mapped file reading for efficient data access
- 🔄 **MPI Support**: Parallel reading with automatic block distribution
- 📊 **Complete Data Access**: Read coordinates, connectivity, and all field variables
- 🎯 **2D/3D Support**: Handles both 2D and 3D meshes seamlessly
- 🔧 **Flexible**: Support for single/double precision and different endianness
- ✅ **Well-Tested**: Comprehensive test suite with 30+ tests

## Installation

### From PyPI

```bash
pip install nek5000reader
```

### From Source

```bash
git clone https://github.com/mvictoras/nek5000reader.git
cd nek5000reader
pip install -e .
```

### Dependencies

- Python 3.9+
- NumPy
- mpi4py
- MPI implementation (OpenMPI, MPICH, etc.)

## Quick Start

### Serial Reading

```python
from nek5000reader import Nek5000Reader

# Initialize reader
reader = Nek5000Reader("simulation.nek5000")

# Get dataset information
info = reader.get_info()
print(f"Number of blocks: {info['num_blocks']}")
print(f"Number of timesteps: {info['num_timesteps']}")

# Read a timestep
data = reader.read_timestep(0)

# Access data
coords = data['coordinates']          # Shape: (num_vertices, 3)
connectivity = data['connectivity']   # 1D array for VTK-style connectivity
velocity = data['fields']['Velocity'] # Flat array: [Vx..., Vy..., Vz...]
pressure = data['fields']['Pressure'] # Shape: (num_vertices,)
```

### MPI Parallel Reading

```python
from mpi4py import MPI
from nek5000reader import Nek5000Reader

# Initialize with MPI communicator
comm = MPI.COMM_WORLD
reader = Nek5000Reader("simulation.nek5000", comm=comm)

# Each rank reads its portion of the data
data = reader.read_timestep(0)

# Process local data
local_coords = data['coordinates']
print(f"Rank {comm.Get_rank()}: {local_coords.shape[0]} vertices")
```

### Reading Multiple Timesteps

```python
reader = Nek5000Reader("simulation.nek5000")

# Get list of available timesteps
timesteps = reader.get_timestep_list()

# Read specific range (start:end:stride)
timesteps = reader.get_timestep_list("0:100:10")  # Every 10th timestep

# Process all timesteps
for step in timesteps:
    data = reader.read_timestep(step)
    # Process data...
```

## Testing

The test suite includes both serial and MPI tests.

### Run Serial Tests

```bash
pytest tests/test_reader.py -v
```

### Run MPI Tests

```bash
# With 2 MPI ranks
mpiexec -n 2 pytest tests/mpi/test_mpi_reader.py -v

# With 4 MPI ranks
mpiexec -n 4 pytest tests/mpi/test_mpi_reader.py -v
```

### Run All Tests with Coverage

```bash
# Serial tests with coverage
pytest tests/test_reader.py -v --cov=nek5000reader --cov-report=html

# MPI tests with coverage
mpiexec -n 2 pytest tests/mpi/test_mpi_reader.py -v --cov=nek5000reader
```

## API Reference

### Nek5000Reader

Main class for reading Nek5000 files.

**Methods:**

- `__init__(nek5000_file, comm=None)`: Initialize reader
- `read_timestep(step, read_mesh=True)`: Read a single timestep
- `get_timestep_list(step_range=None)`: Get list of timesteps to process
- `get_info()`: Get dataset information

**Returns from `read_timestep()`:**

```python
{
    'coordinates': np.ndarray,      # (num_vertices, 3)
    'connectivity': np.ndarray,     # 1D array
    'fields': {
        'Velocity': np.ndarray,     # Flat: [Vx..., Vy..., Vz...]
        'Velocity Magnitude': np.ndarray,
        'Pressure': np.ndarray,
        'Temperature': np.ndarray,
        # ... other fields
    },
    'time': float,
    'cycle': int,
    'metadata': dict
}
```

## File Format

The reader expects:
- A `.nek5000` control file containing metadata
- Corresponding data files (e.g., `prefix0.f00000`, `prefix0.f00001`, ...)

Example `.nek5000` file:
```
filetemplate: prefix%01d.f%05d
firsttimestep: 0
numtimesteps: 100
```

## Performance Tips

1. **Use MPI for large datasets**: Distribute blocks across ranks for parallel I/O
2. **Skip mesh reading when not needed**: Use `read_mesh=False` for subsequent timesteps
3. **Memory-mapped I/O**: The reader uses `np.memmap` for efficient data access
4. **Batch processing**: Process multiple timesteps in a single run

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

BSD 3-Clause License. See [LICENSE](LICENSE) for details.

## Citation

If you use this library in your research, please cite:

```bibtex
@software{nek5000reader,
  author = {Mateevitsi, Victor},
  title = {Nek5000 Reader: A Python Library for Nek5000 Data},
  year = {2025},
  url = {https://github.com/mvictoras/nek5000reader}
}
```

## Acknowledgments

This library was developed for efficient post-processing of Nek5000 CFD simulations.

### Test Data

Test data files in `tests/data/` are from the [pymech-test-data](https://github.com/eX-Mech/pymech-test-data) repository (GPL-3.0 licensed). See `tests/data/ATTRIBUTION.md` for details.


