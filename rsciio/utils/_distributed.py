# -*- coding: utf-8 -*-
# Copyright 2007-2026 The HyperSpy developers
#
# This file is part of RosettaSciIO.
#
# RosettaSciIO is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# RosettaSciIO is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with RosettaSciIO. If not, see <https://www.gnu.org/licenses/#GPL>.

import os
import sys

import dask.array as da
import numpy as np


def get_chunk_slice(
    shape,
    chunks="auto",
    block_size_limit=None,
    dtype=None,
):
    """
    Get chunk slices for the :func:`rsciio.utils.distributed.slice_memmap` function.

    Takes a shape and chunks and returns a dask array of the slices to be used with the
    :func:`rsciio.utils.distributed.slice_memmap` function. This is useful for loading data
    from a memmaped file in a distributed manner.

    Parameters
    ----------
    shape : tuple
        Shape of the data.
    chunks : tuple or str, optional
        Define the chunk shape. This argument is passed to :func:`dask.array.core.normalize_chunks`.
        The default is "auto".
    block_size_limit : int, optional
        Maximum size of a block in bytes. The default is None. This is passed
        to the :py:func:`dask.array.core.normalize_chunks` function when chunks == "auto".
    dtype : numpy.dtype, optional
        Data type. The default is None. This is passed to the
        :py:func:`dask.array.core.normalize_chunks` function when chunks == "auto".

    Returns
    -------
    dask.array.Array
        Dask array of the slices.
    tuple
        Tuple of the chunks.
    """

    chunks = da.core.normalize_chunks(
        chunks=chunks, shape=shape, limit=block_size_limit, dtype=dtype
    )
    chunks_shape = tuple([len(c) for c in chunks])
    slices = np.empty(
        shape=chunks_shape + (len(chunks_shape), 2),
        dtype=int,
    )
    for ind in np.ndindex(chunks_shape):
        current_chunk = [chunk[i] for i, chunk in zip(ind, chunks)]
        starts = [int(np.sum(chunk[:i])) for i, chunk in zip(ind, chunks)]
        stops = [s + c for s, c in zip(starts, current_chunk)]
        slices[ind] = [[start, stop] for start, stop in zip(starts, stops)]

    return da.from_array(slices, chunks=(1,) * len(shape) + slices.shape[-2:]), chunks


def get_arbitrary_chunk_slice(
    positions,
    shape,
    chunks="auto",
    block_size_limit=None,
    dtype=None,
):
    """
    Get chunk slices for the :func:`rsciio.utils.distributed.slice_memmap` function. From arbitrary positions
    given by a list of x, y coordinates.

    Parameters
    ----------
    positions : array-like
        A numpy array in the form [[x1, y1], [x2, y2], ...] where x, y map the frame to the
        real space coordinate of the data.
    shape : tuple
        Shape of the signal data.
    chunks : tuple, optional
        Chunk shape. The default is "auto".
    block_size_limit : int, optional
        Maximum size of a block in bytes. The default is None. This is passed
        to the :py:func:`dask.array.core.normalize_chunks` function when chunks == "auto".
    dtype : numpy.dtype, optional
        Data type. The default is None. This is passed to the
        :py:func:`dask.array.core.normalize_chunks` function when chunks == "auto".

    Returns
    -------
    dask.array.Array
        Dask array of the slices.
    """
    if not isinstance(positions, np.ndarray):
        positions = np.array(positions)
    if chunks == "auto":
        chunks = ("auto",) * (len(shape) - 2) + (-1, -1)
    elif chunks[-2:] != (-1, -1):
        raise ValueError("Last two dimensions of chunks must be -1")
    chunks = da.core.normalize_chunks(
        chunks=chunks, shape=shape, limit=block_size_limit, dtype=dtype
    )
    pos_mapping = np.zeros(shape=shape[:-2] + (1, 1), dtype=int)

    for i, p in enumerate(positions):
        pos_mapping[tuple(p)] = i + 1
    pos_mapping = pos_mapping - 1  # 0 based indexing, -1 for the empty frames

    # Now we chunk the pos_mapping array.  In the case each frame remains in a single chunk and we only
    # return the navigation dimensions.  Later when we populate the data we will use the pos_mapping array
    # map some frame index to the position within a dense array.
    return da.from_array(pos_mapping, chunks=chunks[:-2] + (1, 1)), chunks


# ---------------------------------------------------------------------------
# Windows sequential-read helper
# ---------------------------------------------------------------------------
# Benchmarking showed that FILE_FLAG_NO_BUFFERING (unbuffered) is NOT the
# right tool for sequential RAID reads on Windows.  The OS page cache with
# FILE_FLAG_SEQUENTIAL_SCAN issues large read-ahead requests that saturate
# all RAID stripes simultaneously, reaching ~2 GB/s.  Unbuffered I/O
# bypasses this prefetching and stalls at ~700-900 MB/s because each ReadFile
# call must wait for a full round-trip before the next one begins.
#
# The correct approach is buffered I/O opened with FILE_FLAG_SEQUENTIAL_SCAN
# so Windows knows to prefetch aggressively.  Python's built-in open() uses
# this flag on Windows when reading sequentially, so we open a fresh file
# handle per chunk and use readinto() into a pre-allocated numpy buffer.
# This avoids the extra copy that f.read() → bytes → np.frombuffer would
# incur, and matches the ~2 GB/s achieved by np.memmap on large sequential
# reads while adding no per-call open() overhead (0.1 ms vs 150-200 ms of
# actual I/O per 128 MB chunk).


def _slice_sequential(
    file: str,
    dtype: np.dtype,
    full_shape: tuple,
    byte_offset: int,
    slices_: np.ndarray,
    order: str,
) -> np.ndarray:
    """
    Read a chunk slice from ``file`` using buffered sequential reads
    (open + seek + readinto), reading ONLY the bytes the slice covers.

    For C-order data every chunk is a union of contiguous byte runs: the
    trailing axes that are read in full extent (plus the last sliced axis,
    whose contiguous index range is also contiguous on disk) form one run;
    the outer sliced axes are iterated, one seek+readinto per run. This reads
    exactly the requested bytes — never the whole file — so a 4-D chunk that
    is chunked on BOTH navigation axes (e.g. a non-square scan's
    ``(64, 64, ky, kx)`` chunk) is read distributed-style instead of mapping
    the full multi-GB file (which fails on Windows with WinError 8).

    F-order is not contiguous-friendly this way → fall back to ``numpy.memmap``
    of the requested region only.
    """
    ndim = len(full_shape)
    itemsize = dtype.itemsize
    full_shape = tuple(int(s) for s in full_shape)
    result_shape = tuple(int(slices_[i, 1] - slices_[i, 0]) for i in range(ndim))

    if order != "C":
        # F-order: bounded memmap of just the leading-axis band, then slice the
        # inner dims (never maps the whole file).
        mm = np.memmap(
            file, dtype=dtype, shape=full_shape, mode="r",
            offset=byte_offset, order=order,
        )
        return np.array(mm[tuple(slice(int(s[0]), int(s[1])) for s in slices_)])

    # Per-axis element stride in C-order (items, not bytes).
    elem_strides = [1] * ndim
    for i in range(ndim - 2, -1, -1):
        elem_strides[i] = elem_strides[i + 1] * full_shape[i + 1]

    # The contiguous run covers the LAST axis that is actually sliced (i.e. not
    # full extent) plus every axis after it (all full extent). `run_axis` is the
    # first axis from the end whose slice is the full extent breaks the search;
    # we want the last NON-full axis. Default: only the innermost run when all
    # axes are full (whole array) or the leading axis is the only sliced one.
    run_axis = 0
    for i in range(ndim):
        if not (slices_[i, 0] == 0 and int(slices_[i, 1]) == full_shape[i]):
            run_axis = i  # last sliced (non-full) axis seen so far
    # Run length in items: from run_axis (its sliced extent) through the end.
    run_items = result_shape[run_axis] * int(np.prod(full_shape[run_axis + 1:])) \
        if run_axis + 1 <= ndim else result_shape[run_axis]
    run_items = int(run_items)

    out = np.empty(result_shape, dtype=dtype, order="C")
    out_flat = out.reshape(-1)

    # Iterate the OUTER axes (those before run_axis); each combination is one
    # contiguous run on disk and one contiguous block in the output.
    outer_axes = range(run_axis)
    outer_ranges = [range(int(slices_[a, 0]), int(slices_[a, 1])) for a in outer_axes]
    out_block = 0
    with open(file, "rb") as fh:
        if not outer_ranges:
            # Single contiguous run (e.g. leading axis only, or whole array).
            start_elem = sum(int(slices_[a, 0]) * elem_strides[a] for a in range(ndim))
            fh.seek(byte_offset + start_elem * itemsize)
            fh.readinto(out_flat)
        else:
            import itertools as _it
            for combo in _it.product(*outer_ranges):
                start_elem = 0
                for a, idx in zip(outer_axes, combo):
                    start_elem += idx * elem_strides[a]
                # add the offset of the run_axis slice start and deeper full axes
                start_elem += int(slices_[run_axis, 0]) * elem_strides[run_axis]
                fh.seek(byte_offset + start_elem * itemsize)
                fh.readinto(out_flat[out_block:out_block + run_items])
                out_block += run_items

    return out


# ---------------------------------------------------------------------------
# slice_memmap -- dispatches to sequential read path when _sequential_read=True
# ---------------------------------------------------------------------------


def slice_memmap(slices, file, dtypes, shape, key=None, positions=False, **kwargs):
    """
    Slice a memory mapped file using a tuple of slices.

    This is useful for loading data from a memory mapped file in a distributed manner. The function
    first creates a memory mapped array of the entire dataset and then uses the ``slices`` to slice the
    memory mapped array.  The slices can be used to build a ``dask`` array as each slice translates to one
    chunk for the ``dask`` array.

    :func:`binary_read_distributed` injects ``_sequential_read=True`` into
    kwargs so that each chunk is read via ``open() + seek + readinto()``
    instead of ``numpy.memmap``.  This avoids creating a mapping of the full
    file per chunk call and uses buffered sequential I/O with OS read-ahead.
    The flag is consumed here and never forwarded to ``numpy.memmap``, so
    existing callers that do not pass it are unaffected.

    Parameters
    ----------
    slices : array-like of int
        An array of the slices to use. The shape of the array should be (n, 2)
        where n is the number of dimensions of the data. The first column is
        the start of the slice and the second column is the stop of the slice.
    file : str
        Path to the file.
    dtypes : numpy.dtype
        Data type of the data for :class:`numpy.memmap` function.
    shape : tuple
        Shape of the entire dataset. Passed to the :class:`numpy.memmap` function.
    key : None, str
        For structured dtype only. Specify the key of the structured dtype to use.
    positions : bool, optional
        If True, the slices include indexes for positions which are then used to
        create a custom scan pattern. The default is False.
    **kwargs : dict
        Additional keyword arguments to pass to the :class:`numpy.memmap` function.

    Returns
    -------
    numpy.ndarray
        Array of the data from the memory mapped file sliced using the provided slice.
    """
    use_sequential = kwargs.pop("_sequential_read", False)
    slices_ = np.squeeze(slices)[()]

    if use_sequential and not positions and key is None:
        byte_offset = kwargs.get("offset", 0)
        order = kwargs.get("order", "C")
        if slices_.ndim == 1:
            slices_2d = np.array([[slices_[0], slices_[1]]])
            result = _slice_sequential(file, dtypes, shape, byte_offset, slices_2d, order)
            return result.reshape(int(slices_[1] - slices_[0]))
        return _slice_sequential(file, dtypes, shape, byte_offset, slices_, order)

    # Standard path: np.memmap (structured dtype, positions, or explicit fallback)
    data = np.memmap(file, dtypes, shape=shape, **kwargs)
    if key is not None:
        data = data[key]
    if positions:
        # We have arbitrary positions.
        if -1 in slices_:  # -1 means empty frame we will return 0.
            result = data[slices_]
            result[slices_ == -1] = 0
            return result
        else:
            return data[slices_]
    else:
        if slices_.ndim == 1:
            # Special case data with single axis
            slices_ = slice(*tuple(slices_))
        else:
            slices_ = tuple([slice(s[0], s[1]) for s in slices_])
        return data[slices_]


def memmap_distributed(
    filename,
    dtype,
    positions=None,
    offset=0,
    shape=None,
    order="C",
    chunks="auto",
    block_size_limit=None,
    key=None,
):
    """
    Drop in replacement for :class:`numpy.memmap` allowing for distributed
    loading of data.

    This always loads the data using dask which can be beneficial in many
    cases, but may not be ideal in others. The ``chunks`` and ``block_size_limit``
    are for describing an ideal chunk shape and size as defined using the
    :func:`dask.array.core.normalize_chunks` function.

    Parameters
    ----------
    filename : str
        Path to the file.
    dtype : numpy.dtype
        Data type of the data for memmap function.
    positions : array-like, optional
        A numpy array in the form [[x1, y1], [x2, y2], ...] where x, y map the frame to the
        real space coordinate of the data. The default is None.
    offset : int, optional
        Offset in bytes. The default is 0.
    shape : tuple, optional
        Shape of the data to be read. The default is None.
    order : str, optional
        Order of the data. The default is "C" see :class:`numpy.memmap` for more details.
    chunks : tuple, optional
        Chunk shape. The default is "auto".
    block_size_limit : int, optional
        Maximum size of a block in bytes. The default is None.
    key : None, str
        For structured dtype only. Specify the key of the structured dtype to use.

    Returns
    -------
    dask.array.Array
        Dask array of the data from the memmaped file and with the specified chunks.

    Notes
    -----
    Currently :func:`dask.array.map_blocks` does not allow for multiple outputs.
    As a result, in case of structured dtype, the key of the structured dtype need
    to be specified.
    For example: with dtype = (("data", int, (128, 128)), ("sec", "<u4", 512)),
    "data" or "sec" will need to be specified.
    """

    if dtype.names is not None:
        # Structured dtype
        array_dtype = dtype[key].base
        sub_array_shape = dtype[key].shape
    else:
        array_dtype = dtype.base
        sub_array_shape = dtype.shape

    if shape is None:
        unit_size = np.dtype(dtype).itemsize
        shape = int(os.path.getsize(filename) / unit_size)
    if not isinstance(shape, tuple):
        shape = (shape,)

    num_dim = len(shape + sub_array_shape)
    if positions is not None:
        # We have arbitrary positions
        chunked_slices, data_chunks = get_arbitrary_chunk_slice(
            positions=positions,
            shape=shape + sub_array_shape,
            chunks=chunks,
            block_size_limit=block_size_limit,
            dtype=array_dtype,
        )
        drop_axes = None
        use_positions = True
        shape = (len(positions),) + shape[-2:]  # update the shape to be linear
    else:
        # Separates slices into appropriately sized chunks.
        chunked_slices, data_chunks = get_chunk_slice(
            shape=shape + sub_array_shape,
            chunks=chunks,
            block_size_limit=block_size_limit,
            dtype=array_dtype,
        )
        drop_axes = (
            num_dim,
            num_dim + 1,
        )  # Dask 2021.10.0 minimum to use negative indexing
        use_positions = False
    data = da.map_blocks(
        slice_memmap,
        chunked_slices,
        file=filename,
        dtype=array_dtype,
        shape=shape,
        order=order,
        mode="r",
        dtypes=dtype,
        offset=offset,
        chunks=data_chunks,
        drop_axis=drop_axes,
        positions=use_positions,
        key=key,
    )
    return data


# Sentinel distinguishing "caller did not pass chunks" from None
_CHUNKS_DEFAULT = object()


def _resolve_chunks_default(shape):
    """
    Compute a sensible chunk default for a binary sequential read.

    Keeps the last two dimensions (signal frame) whole so each dask task
    reads a contiguous block of complete patterns -- avoids read amplification
    from splitting diffraction frames across chunk boundaries.
    """
    ndim = len(shape)
    if ndim >= 3:
        return ("auto",) * (ndim - 2) + (-1, -1)
    elif ndim == 2:
        return ("auto", -1)
    return "auto"


def _build_sequential_graph(filename, dtype, shape, offset, order, chunks,
                             block_size_limit):
    """
    Build a dask graph that reads each chunk via open+seek+readinto.

    Returns a dask.array.Array.  Used by read_binary_distributed when the
    ``"sequential"`` backend is active.
    """
    if dtype.names is not None:
        raise ValueError(
            "Sequential backend does not support structured dtypes. "
            "Use backend='memmap' or pass key= to memmap_distributed."
        )
    array_dtype = dtype.base
    sub_array_shape = dtype.shape
    num_dim = len(shape + sub_array_shape)

    chunked_slices, data_chunks = get_chunk_slice(
        shape=shape + sub_array_shape,
        chunks=chunks,
        block_size_limit=block_size_limit,
        dtype=array_dtype,
    )
    drop_axes = (num_dim, num_dim + 1)

    return da.map_blocks(
        slice_memmap,
        chunked_slices,
        file=filename,
        dtype=array_dtype,
        shape=shape,
        order=order,
        mode="r",
        dtypes=dtype,
        offset=offset,
        chunks=data_chunks,
        drop_axis=drop_axes,
        positions=False,
        key=None,
        _sequential_read=True,
    )


def read_binary_distributed(
    filename,
    dtype,
    positions=None,
    offset=0,
    shape=None,
    order="C",
    chunks=_CHUNKS_DEFAULT,
    block_size_limit=None,
    key=None,
    backend=None,
):
    """
    Distributed lazy reader for binary files with automatic backend selection.

    Drop-in replacement for :func:`memmap_distributed`.  Adds:

    * **Smart chunk default** -- signal frames (last two dims) are kept whole
      so each dask task is a contiguous sequential read rather than a
      fragmented tile.
    * **Pluggable backends** -- choose how each chunk is read from disk.
    * **Automatic machine detection** -- on first use, a short benchmark
      determines the fastest backend for this storage device and caches
      the result in ``platformdirs.user_config_dir("rosettasciio")``.

    Backends
    --------
    ``"memmap"``
        ``numpy.memmap`` -- maps the full file per chunk call.  Portable;
        best when data is already in the OS page cache.
    ``"sequential"``
        ``open() + seek + readinto()`` directly into the output buffer.
        Avoids full-file VMA overhead; OS read-ahead saturates RAID bandwidth.
        Falls back to ``"memmap"`` for structured dtypes, ``positions``, or
        F-order data.
    ``None`` (default)
        Use the system default chosen by
        :func:`rsciio.utils.io_backend.get_default_backend`.

    Parameters
    ----------
    filename : str
        Path to the binary file.
    dtype : numpy.dtype
        Data type.
    positions : array-like, optional
        Custom scan positions (see :func:`memmap_distributed`).
    offset : int, optional
        Byte offset to the start of data.
    shape : tuple, optional
        Shape of the data.  Inferred from file size if ``None``.
    order : str, optional
        ``"C"`` (default) or ``"F"``.
    chunks : tuple or str, optional
        Dask chunk spec.  Default keeps signal frames whole:
        ``("auto",) * (ndim-2) + (-1, -1)``.
    block_size_limit : int, optional
        Max chunk bytes for dask normalisation.
    key : str, optional
        Structured-dtype field to extract.
    backend : str or None, optional
        ``"memmap"``, ``"sequential"``, or ``None`` (auto-detect).

    Returns
    -------
    dask.array.Array

    See Also
    --------
    rsciio.utils.io_backend.get_default_backend
    rsciio.utils.io_backend.set_default_backend
    rsciio.utils.io_backend.benchmark_backends
    """
    from rsciio.utils._io_backend import get_default_backend, BACKENDS

    if backend is None:
        backend = get_default_backend()

    if backend not in BACKENDS:
        raise ValueError(f"Unknown backend {backend!r}. Must be one of {BACKENDS}.")

    # Resolve shape before computing chunk default
    if shape is None:
        unit_size = np.dtype(dtype).itemsize
        shape = int(os.path.getsize(filename) / unit_size)
    if not isinstance(shape, tuple):
        shape = (shape,)
    shape = tuple(int(s) for s in shape)  # ensure plain Python ints

    # "auto" (dask default) splits all dims including signal dims, producing
    # fragmented tile reads.  Treat it the same as our sentinel so we apply
    # the smart sequential default.  Callers that genuinely want dask to
    # choose chunk sizes across all dims can pass chunks=None explicitly.
    if chunks is _CHUNKS_DEFAULT or chunks == "auto":
        chunks = _resolve_chunks_default(shape)

    # Cases that require the memmap path regardless of backend choice:
    # structured dtypes with key, arbitrary positions, F-order
    use_memmap = (
        backend == "memmap"
        or positions is not None
        or key is not None
        or order != "C"
        or (dtype.names is not None)
    )

    if use_memmap:
        return memmap_distributed(
            filename,
            dtype=dtype,
            positions=positions,
            offset=offset,
            shape=shape,
            order=order,
            chunks=chunks,
            block_size_limit=block_size_limit,
            key=key,
        )

    return _build_sequential_graph(
        filename, dtype, shape, offset, order, chunks, block_size_limit
    )
