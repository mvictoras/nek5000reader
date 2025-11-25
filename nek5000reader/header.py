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
Functions for reading Nek5000 file headers and metadata.
"""

import os
import struct
from typing import Tuple, Dict, List, Optional
import numpy as np

from .utils import read_ascii_token, peek, skip_spaces, skip_digits, last_int_in_string


def parse_nek5000_control(path: str) -> Dict:
    """
    Parse the .nek5000 control file for:
      filetemplate: printf-style template (e.g., prefix%05d.fld)
      firsttimestep: int
      numtimesteps: int
      
    Args:
        path: Path to .nek5000 control file
        
    Returns:
        Dictionary with filetemplate, firsttimestep, and numtimesteps
    """
    out = {"filetemplate": None, "firsttimestep": None, "numtimesteps": None}
    with open(path, "r", encoding="utf-8", errors="ignore") as fp:
        toks = fp.read().replace("\r", "\n").split()
    # Very simple tag-based parse
    for i, t in enumerate(toks):
        lt = t.lower()
        if lt.startswith("filetemplate:") and i + 1 < len(toks):
            out["filetemplate"] = toks[i + 1]
        elif lt.startswith("firsttimestep:") and i + 1 < len(toks):
            out["firsttimestep"] = int(toks[i + 1])
        elif lt.startswith("numtimesteps:") and i + 1 < len(toks):
            out["numtimesteps"] = int(toks[i + 1])
    if out["filetemplate"] is None:
        raise RuntimeError("Missing 'filetemplate:' in .nek5000 file.")
    if out["firsttimestep"] is None:
        raise RuntimeError("Missing 'firsttimestep:' in .nek5000 file.")
    if out["numtimesteps"] is None:
        raise RuntimeError("Missing 'numtimesteps:' in .nek5000 file.")
    # Make absolute if necessary
    if not os.path.isabs(out["filetemplate"]):
        out["filetemplate"] = os.path.join(os.path.dirname(path), out["filetemplate"])
    return out


def read_basic_header_and_endian(dfname: str) -> Tuple[int, Tuple[int,int,int], int, bool]:
    """
    Reads the initial ASCII header (#std, precision, blockDims, ..., numBlocks),
    probes endian using the float at offset 132, and returns:
      precision_bytes (4 or 8), (nx, ny, nz), numBlocks, swapEndian(bool)
      
    Args:
        dfname: Path to Nek5000 data file
        
    Returns:
        Tuple of (precision_bytes, (nx, ny, nz), numBlocks, swapEndian)
    """
    with open(dfname, "rb") as f:
        # Tokens: "#std", precision, nx, ny, nz, (some token), numBlocks
        tag = read_ascii_token(f)
        if tag != "#std":
            raise RuntimeError(f"{dfname}: expected '#std' at start, got '{tag}'")
        precision = int(read_ascii_token(f))
        nx = int(read_ascii_token(f))
        ny = int(read_ascii_token(f))
        nz = int(read_ascii_token(f))
        _ = read_ascii_token(f)  # "blocks per file" label or similar
        numBlocks = int(read_ascii_token(f))

        # Probe endian via float at offset 132
        f.seek(132, 0)
        b = f.read(4)
        if len(b) != 4:
            raise RuntimeError("Could not read endian probe.")
        test_le = struct.unpack("<f", b)[0]
        test_be = struct.unpack(">f", b)[0]
        # VTK checks ~6.5..6.6
        def ok(v): return 6.5 < v < 6.6
        if ok(test_le):
            swap = False  # file little-endian matches our unpack
        elif ok(test_be):
            swap = True
        else:
            # Fallback: assume native little endian; still proceed
            swap = False
    return precision, (nx, ny, nz), numBlocks, swap


def read_time_and_tags(dfname: str) -> Tuple[float, int, str, bool]:
    """
    Reads (time, cycle, tags_string, has_mesh) from the ASCII/tag section.
    Falls back to parsing the step from the filename if cycle == 0.
    
    Args:
        dfname: Path to Nek5000 data file
        
    Returns:
        Tuple of (time, cycle, tags_string, has_mesh)
    """
    with open(dfname, "rb") as f:
        # Skip the first 7 tokens, then read time, cycle, one token (dummy),
        # then skip spaces + digits (num directories), then read 32 raw bytes as tags.
        for _ in range(7):
            _ = read_ascii_token(f)
        t_str = read_ascii_token(f)
        c_str = read_ascii_token(f)
        _ = read_ascii_token(f)
        skip_spaces(f)
        skip_digits(f)
        tags = f.read(32)
        tags = (tags or b"").decode("ascii", errors="ignore")
        has_mesh = ("X" in tags)

        # parse
        t = float(t_str) if t_str and any(ch.isdigit() for ch in t_str) else 0.0
        c = int(c_str) if c_str and c_str.strip("-+").isdigit() else 0

    # Fallback: many Nek files leave cycle at 0; use the last integer in the filename.
    if c == 0:
        base = os.path.basename(dfname)
        c_from_name = last_int_in_string(base)
        if c_from_name is not None:
            c = c_from_name

    return t, c, tags, has_mesh


def parse_var_tags(tags: str, mesh_is_3d: bool) -> Tuple[List[str], List[int]]:
    """
    From the tags string, produce var_names and component counts.
    Adds "Velocity Magnitude" right after "Velocity" to mirror VTK.
    
    Args:
        tags: Tags string from file header
        mesh_is_3d: Whether mesh is 3D
        
    Returns:
        Tuple of (variable_names, component_counts)
    """
    names: List[str] = []
    lens: List[int] = []
    # Count S fields
    s_count = 0
    if "S" in tags:
        # Find 'S##' (two digits) after 'S', else default to 1
        # We scan the next two chars that look like digits:
        idx = tags.find("S")
        if idx >= 0 and idx + 2 < len(tags):
            d = "".join([c for c in tags[idx+1:idx+3] if c.isdigit()])
            s_count = int(d) if len(d) == 2 else 1
        else:
            s_count = 1
    # Velocity
    if "U" in tags:
        names.append("Velocity")
        lens.append(3 if mesh_is_3d else 3)  # we keep 3 comps; Z set to 0 in 2D
        names.append("Velocity Magnitude")
        lens.append(1)
    # Pressure
    if "P" in tags:
        names.append("Pressure")
        lens.append(1)
    # Temperature
    if "T" in tags:
        names.append("Temperature")
        lens.append(1)
    # Scalars S01...SNN
    for s in range(s_count):
        names.append(f"S{s+1:02d}")
        lens.append(1)
    return names, lens


def read_block_ids(dfname: str, numBlocks: int, swapEndian: bool) -> np.ndarray:
    """
    Read block IDs from file.
    
    Args:
        dfname: Path to Nek5000 data file
        numBlocks: Number of blocks
        swapEndian: Whether to swap byte order
        
    Returns:
        Array of block IDs
    """
    with open(dfname, "rb") as f:
        f.seek(136, 0)  # block id list starts at 136
        arr = np.fromfile(f, dtype=np.int32, count=numBlocks)
    if arr.size != numBlocks:
        raise RuntimeError("Failed to read block IDs.")
    if swapEndian:
        arr.byteswap(inplace=True)
    return arr


def read_map_file(nekfile: str) -> Optional[np.ndarray]:
    """
    If a .map file exists alongside the .nek5000, read its element order.
    File format: first int: num_map_elements, followed by per-line entries
    where the first numeric field is the element id (0-based) -> +1 like VTK.
    
    Args:
        nekfile: Path to .nek5000 file
        
    Returns:
        Array of element IDs, or None if no map file exists
    """
    map_path = os.path.splitext(nekfile)[0] + ".map"
    if not os.path.exists(map_path):
        return None
    ids = []
    with open(map_path, "r", encoding="utf-8", errors="ignore") as fp:
        toks = fp.read().split()
    if not toks:
        return None
    # toks: num_elems then repeated lines; element id is first int on each line
    num_elems = int(toks[0])
    # Heuristic: the next 8 tokens per element; first is id
    # Safer approach: scan lines ignoring text; grab first int on each line.
    with open(map_path, "r", encoding="utf-8", errors="ignore") as fp:
        first = True
        for line in fp:
            if first:
                first = False
                continue
            parts = line.strip().split()
            if not parts:
                continue
            try:
                ids.append(int(parts[0]) + 1)  # VTK adds +1
            except ValueError:
                continue
    if len(ids) != num_elems:
        # fallback: ignore map
        return None
    return np.array(ids, dtype=np.int32)


def total_header_size_bytes(numBlocks: int, totalBlockSize: int, comps_xyz: int, 
                            precision: int, has_mesh: bool) -> int:
    """
    Calculate total header size in bytes.
    
    Args:
        numBlocks: Number of blocks
        totalBlockSize: Size of each block
        comps_xyz: Number of coordinate components
        precision: Precision in bytes (4 or 8)
        has_mesh: Whether mesh data is present
        
    Returns:
        Header size in bytes
    """
    base = 136 + numBlocks * 4  # 136 header + block id table
    if has_mesh:
        base += numBlocks * totalBlockSize * comps_xyz * precision
    return base
