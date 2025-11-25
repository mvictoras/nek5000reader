# Test Data Attribution

## stsabl Test Data

The test data files in `tests/data/` with the prefix `stsabl` are from the [pymech-test-data](https://github.com/eX-Mech/pymech-test-data) repository:

- Repository: https://github.com/eX-Mech/pymech-test-data
- License: GPL-3.0

### Files

- `stsabl.nek5000` - Nek5000 control file
- `stsabl0.f00000` - Nek5000 field data file

These files are used for testing purposes only and are subject to the GPL-3.0 license from the original repository.

## License Compatibility

The test data files are licensed under GPL-3.0, while the nek5000reader library itself is licensed under BSD-3-Clause. This is compatible as:

1. The test data is only used for testing and is not distributed as part of the library package
2. The test data is clearly attributed and its license is documented
3. The library code itself remains BSD-3-Clause licensed

Users of the nek5000reader library are not required to comply with GPL-3.0 unless they choose to use or redistribute the test data files.
