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
Utility functions for Nek5000 file handling.
"""

import re
from typing import Optional, Tuple

import numpy as np
from mpi4py import MPI


def build_step_filename(fmt: str, step: int, dir_index: int = 0) -> str:
    """
    Supports Nek5000 templates with one or two printf specifiers.
      e.g. "data%05d.fld"            -> fmt % step
            "turbPipe%01d.f%05d"     -> fmt % (dir_index, step)  (VTK passes 0 for dir_index)
    Falls back to simple concatenation if there are no specifiers.

    Args:
        fmt: Printf-style format string
        step: Timestep number
        dir_index: Directory index (default: 0)

    Returns:
        Formatted filename string
    """
    if "%" not in fmt:
        return f"{fmt}{step}"
    # try single-arg first
    try:
        return fmt % step
    except TypeError:
        pass
    # try (dir_index, step) like VTK does
    try:
        return fmt % (dir_index, step)
    except TypeError as e:
        raise RuntimeError(
            f"Unsupported filetemplate '{fmt}'. Expected 0, 1, or 2 printf specifiers."
        ) from e


def partition_blocks(numBlocks: int, comm: MPI.Comm) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (counts_per_rank, displs) for an even partition of elements.

    Args:
        numBlocks: Total number of blocks to partition
        comm: MPI communicator

    Returns:
        Tuple of (counts_per_rank, displacements) as numpy arrays
    """
    size = comm.Get_size()
    base = numBlocks // size
    rem = numBlocks % size
    counts = np.array(
        [base + (1 if r < rem else 0) for r in range(size)], dtype=np.int32
    )
    displs = np.zeros(size, dtype=np.int32)
    if size > 1:
        displs[1:] = np.cumsum(counts[:-1])
    return counts, displs


def last_int_in_string(s: str) -> Optional[int]:
    """
    Extract the last integer found in a string.

    Args:
        s: Input string

    Returns:
        Last integer found, or None if no integers present
    """
    m = re.findall(r"(\d+)", s)
    return int(m[-1]) if m else None


def read_ascii_token(f) -> str:
    """
    Read a whitespace-delimited ASCII token from a binary file.

    Args:
        f: File object opened in binary mode

    Returns:
        Decoded ASCII token string
    """
    # Skip whitespace
    ch = f.read(1)
    while ch and ch in b" \t\r\n":
        ch = f.read(1)
    if not ch:
        return ""
    # Read until whitespace
    buf = [ch]
    ch = f.read(1)
    while ch and ch not in b" \t\r\n":
        buf.append(ch)
        ch = f.read(1)
    return b"".join(buf).decode("ascii", errors="ignore")


def peek(f) -> int:
    """
    Peek at the next byte without advancing file position.

    Args:
        f: File object opened in binary mode

    Returns:
        Next byte value, or -1 if at EOF
    """
    pos = f.tell()
    b = f.read(1)
    f.seek(pos)
    return b[0] if b else -1


def skip_spaces(f):
    """Skip whitespace characters in binary file."""
    b = peek(f)
    while b == ord(" "):
        f.read(1)
        b = peek(f)


def skip_digits(f):
    """Skip digit characters in binary file."""
    b = peek(f)
    while b >= ord("0") and b <= ord("9"):
        f.read(1)
        b = peek(f)
