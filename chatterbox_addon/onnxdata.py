"""Read a graph's weights (initializers) straight from its .onnx file.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

The Multilingual export's embedding graph is four lookup tables and one
vector. The add-on reads them as numpy arrays and does the lookups itself,
because the unconditional half of classifier-free guidance needs the text
tables without the text: a graph that adds them together cannot give that.

Only what those tables need is parsed from ONNX's protobuf: a tensor's name,
shape and type, and its bytes, inline or in the external data file beside the
graph. The external data is memory-mapped, not read.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# TensorProto.DataType values this reader takes.
_DTYPES = {1: np.float32, 6: np.int32, 7: np.int64, 10: np.float16, 11: np.float64}


def _varint(data: bytes, pos: int) -> tuple[int, int]:
    shift = result = 0
    while True:
        byte = data[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, pos
        shift += 7


def _fields(data: bytes):
    """(field number, wire type, value) for each field of one message.

    The value is an int for varints and fixed-width fields, a memoryview for
    length-delimited ones.
    """
    view = memoryview(data)
    pos, end = 0, len(data)
    while pos < end:
        key, pos = _varint(data, pos)
        number, wire = key >> 3, key & 7
        if wire == 0:
            value, pos = _varint(data, pos)
        elif wire == 1:
            value, pos = int.from_bytes(data[pos:pos + 8], 'little'), pos + 8
        elif wire == 2:
            length, pos = _varint(data, pos)
            value, pos = view[pos:pos + length], pos + length
        elif wire == 5:
            value, pos = int.from_bytes(data[pos:pos + 4], 'little'), pos + 4
        else:
            raise ValueError(f'unsupported protobuf wire type {wire}')
        yield number, wire, value


def _tensor(message: bytes, folder: Path) -> tuple[str, np.ndarray]:
    name, dims, dtype, raw, external = '', [], 0, None, {}
    for number, wire, value in _fields(message):
        if number == 1:                       # dims, packed or not
            if wire == 2:
                packed, pos = bytes(value), 0
                while pos < len(packed):
                    dim, pos = _varint(packed, pos)
                    dims.append(dim)
            else:
                dims.append(value)
        elif number == 2:
            dtype = value
        elif number == 8:
            name = bytes(value).decode('utf-8')
        elif number == 9:
            raw = bytes(value)
        elif number == 13:                    # external_data: key/value pairs
            entry = dict((n, bytes(v).decode('utf-8')) for n, _, v in _fields(bytes(value)))
            external[entry.get(1, '')] = entry.get(2, '')
    if dtype not in _DTYPES:
        raise ValueError(f'{name}: unsupported tensor type {dtype}')
    numpy_type = np.dtype(_DTYPES[dtype]).newbyteorder('<')
    count = int(np.prod(dims)) if dims else 1
    if external:
        offset = int(external.get('offset', 0))
        array = np.memmap(folder / external['location'], dtype=numpy_type, mode='r',
                          offset=offset, shape=(count,))
    elif raw is not None:
        array = np.frombuffer(raw, dtype=numpy_type, count=count)
    else:
        raise ValueError(f'{name}: no raw or external data')
    return name, array.reshape(dims)


def initializers(path: Path) -> dict[str, np.ndarray]:
    """Every initializer of the graph at `path`, by name."""
    path = Path(path)
    model = path.read_bytes()
    graph = next((bytes(v) for n, _, v in _fields(model) if n == 7), None)
    if graph is None:
        raise ValueError(f'{path}: no graph')
    return dict(_tensor(bytes(v), path.parent) for n, _, v in _fields(graph) if n == 5)
