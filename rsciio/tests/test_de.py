import dask.array
import matplotlib.pyplot as plt
import pytest
from hyperspy.io import load
import numpy as np
import glob
from rsciio.de.api import SeqReader,CeleritasReader,file_reader


class TestShared:
    @pytest.fixture
    def seq(self):
        return SeqReader(file="de_data/data/test.seq",
                         dark="de_data/data/test.seq.dark.mrc",
                         gain="de_data/data/test.seq.gain.mrc",
                         metadata="de_data/data/test.seq.metadata",
                         xml="de_data/data/test.seq.se.xml")

    def test_parse_header(self, seq):
        header = seq._read_file_header()
        assert header["ImageWidth"] == 64
        assert header["ImageHeight"] == 64
        assert header["ImageBitDepthReal"] == 12
        assert header["NumFrames"] == 10
        assert header["TrueImageSize"] == 16384
        np.testing.assert_almost_equal(header["FPS"], 30, 1)  # Note this value is very wrong for Celeritas Camera
        # Read from the xml file...

    def test_parse_metadata(self, seq):
        metadata = seq._read_metadata()
        print(metadata)

    def test_read_dark(self, seq):
        dark, gain = seq._read_dark_gain()
        assert dark.shape == (64, 64)
        assert gain is None

    @pytest.mark.parametrize("nav_shape", [None, (5, 2), (5, 3)])
    def test_read(self, seq, nav_shape):
        data = seq.read_data(navigation_shape=nav_shape)
        if nav_shape is None:
            nav_shape = (10,)
        assert data["data"].shape == (*nav_shape, 64, 64)


class TestLoadCeleritas:
    @pytest.fixture
    def seq(self):
        files = sorted(glob.glob("de_data/celeritas_data/128x256_PRebuffer128/*"))
        kws = {"file":"de_data/celeritas_data/128x256_PRebuffer128/test.seq",
               "top": files[6],
               "bottom":files[4],
               "dark": files[2],
               "gain": files[3],
               "xml": files[0],
               "metadata": files[7]}
        return CeleritasReader(**kws)

    def test_parse_header(self, seq):
        print(seq.bottom)
        header = seq._read_file_header()
        assert header["ImageWidth"] == 256
        assert header["ImageHeight"] == 8192
        assert header["ImageBitDepthReal"] == 12
        assert header["NumFrames"] == 4 #this is wrong
        assert header["TrueImageSize"] == 4202496
        np.testing.assert_almost_equal(header["FPS"], 300, 1)  # This value is wrong for the celeritas camera

    def test_parse_xml(self, seq):
        xml = seq._read_xml()
        assert xml["ImageSizeX"] == 256
        assert xml["ImageSizeY"] == 128
        assert xml["FrameRate"] == 40000 # correct FPS
        assert xml["DarkRef"] == "Yes"
        assert xml["GainRef"] == "Yes"
        assert xml["SegmentPreBuffer"] == 128

    @pytest.mark.parametrize("nav_shape", [None, (50, 200), (5, 3)])
    def test_read(self, seq, nav_shape):
        data_dict = seq.read_data()
        assert data_dict["data"].shape == (128, 128, 256)


def test_load_file():
    data_dict = file_reader("de_data/celeritas_data/128x256_PRebuffer128/test_Top_14-04-59.355.seq",
                            celeritas=True)
    assert data_dict["data"].shape == (128, 128, 256)

def test_load_file2():
    data_dict = file_reader("de_data/celeritas_data/256x256_Prebuffer1/Movie_00785_Top_13-49-04.160.seq",
                            celeritas=True)
    assert data_dict["data"].shape == (5, 256, 256)

def test_load_file3():
    data_dict = file_reader("de_data/celeritas_data/64x64_Prebuffer256/test_Bottom_14-13-42.822.seq",
                            celeritas=True, lazy=True)
    print(data_dict["data"])
    assert isinstance(data_dict["data"],dask.array.Array)
    assert data_dict["data"].shape == (256, 64, 64)

def test_load_file3():
    data_dict = file_reader("de_data/celeritas_data/64x64_Prebuffer256/test_Bottom_14-13-42.822.seq",
                            celeritas=True, lazy=True, navigation_shape=(50,50))
    print(data_dict["data"])
    assert isinstance(data_dict["data"], dask.array.Array)
    assert data_dict["data"].shape == (50, 50, 256, 64, 64)