import numpy as np
import pytest

from rsciio.utils._distributed import get_chunk_slice, memmap_distributed


@pytest.mark.parametrize(
    "shape", ((10, 20, 30, 512, 512), (20, 30, 512, 512), (10, 512, 512), (512, 512))
)
def test_get_chunk_slice(shape):
    chunk_arr, chunk = get_chunk_slice(shape=shape, chunks=-1)  # 1 chunk
    assert chunk_arr.shape == (1,) * len(shape) + (len(shape), 2)
    assert chunk == tuple([(i,) for i in shape])

    chunks = (1,) * (len(shape) - 2) + (-1, -1)
    # Everything is 1 chunk
    chunk_arr, chunk = get_chunk_slice(shape=shape, chunks=chunks)
    assert chunk_arr.shape == shape[:-2] + (1, 1) + (len(shape), 2)
    assert chunk == (
        tuple([(1,) * i for i in shape[:-2]]) + tuple([(i,) for i in shape[-2:]])
    )


def test_memmap_distributed_does_not_open_file_at_build_time(tmp_path, monkeypatch):
    """Building the dask graph must not touch the real file.

    Without an explicit ``meta=`` in the ``map_blocks`` call, dask infers
    output metadata by actually calling the mapped function (``compute_meta``),
    which used to open a real ``np.memmap`` on the target file during graph
    construction -- i.e. during ``file_reader(..., lazy=True)`` itself, before
    any data was requested. On slow/network filesystems this made ``hs.load``
    slow even for lazy loading. See GH discussion on rsciio 0.14.0 slowdown.
    """
    shape = (4, 5, 6, 7)
    dtype = np.dtype("float32")
    data = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
    path = tmp_path / "data.bin"
    path.write_bytes(data.tobytes())

    real_memmap = np.memmap
    opens = []
    monkeypatch.setattr(
        np,
        "memmap",
        lambda *a, **k: opens.append(1) or real_memmap(*a, **k),
    )

    dask_array = memmap_distributed(str(path), dtype=dtype, shape=shape, chunks="auto")
    assert opens == [], "building the dask graph must not open the real file"

    result = dask_array.compute()
    assert len(opens) == 1, "computing should open the file exactly once"
    np.testing.assert_array_equal(result, data)
