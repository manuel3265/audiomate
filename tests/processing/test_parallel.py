import os

import h5py
import numpy as np

import pytest

from audiomate import processing

from tests import resources


class ProcessorDummy(processing.Processor):

    def __init__(self, frame_size_scale=1.0, hop_size_scale=1.0):
        self.called_with_data = []
        self.called_with_sr = []
        self.called_with_offset = []
        self.called_with_last = []
        self.called_with_utterance = []
        self.called_with_corpus = []

        self.mock_frame_size_scale = frame_size_scale
        self.mock_hop_size_scale = hop_size_scale

    def process_frames(self, data, sampling_rate, offset=0, last=False, utterance=None, corpus=None):
        self.called_with_data.append(data)
        self.called_with_sr.append(sampling_rate)
        self.called_with_offset.append(offset)
        self.called_with_last.append(last)
        self.called_with_utterance.append(utterance)
        self.called_with_corpus.append(corpus)

        return data

    def frame_transform(self, frame_size, hop_size):
        tf_frame_size = self.mock_frame_size_scale * frame_size
        tf_hop_size = self.mock_hop_size_scale * hop_size

        return tf_frame_size, tf_hop_size


@pytest.fixture()
def processor():
    return ProcessorDummy()


class TestProcessor:

    def test_process_corpus(self, processor, tmpdir):
        ds = resources.create_dataset()
        feat_path = os.path.join(tmpdir.strpath, 'feats')

        par_processor = processing.ParallelProcessor(processor)
        par_processor.process_corpus(
            ds,
            feat_path,
            4,
            frame_size=4096,
            hop_size=2048
        )

        with h5py.File(feat_path, 'r') as f:
            utts = set(f.keys())

            assert utts == set(ds.utterances.keys())

            assert f['utt-1'].shape == (20, 4096)
            assert f['utt-2'].shape == (20, 4096)
            assert f['utt-3'].shape == (11, 4096)
            assert f['utt-4'].shape == (7, 4096)
            assert f['utt-5'].shape == (20, 4096)

    def test_process_corpus_same_result_as_serial(self, processor, tmpdir):
        ds = resources.create_dataset()
        serial_path = os.path.join(tmpdir.strpath, 'feats_ser')
        parallel_path = os.path.join(tmpdir.strpath, 'feats_par')

        par_processor = processing.ParallelProcessor(processor)
        par_processor.process_corpus(
            ds,
            parallel_path,
            4,
            frame_size=4096,
            hop_size=2048,
            sr=16000
        )

        processor.process_corpus(
            ds,
            serial_path,
            frame_size=4096,
            hop_size=2048,
            sr=16000
        )

        feats_ser = h5py.File(serial_path, 'r')
        feats_par = h5py.File(parallel_path, 'r')

        assert np.array_equal(feats_ser['utt-1'][()], feats_par['utt-1'][()])
        assert np.array_equal(feats_ser['utt-2'][()], feats_par['utt-2'][()])
        assert np.array_equal(feats_ser['utt-3'][()], feats_par['utt-3'][()])
        assert np.array_equal(feats_ser['utt-4'][()], feats_par['utt-4'][()])
        assert np.array_equal(feats_ser['utt-5'][()], feats_par['utt-5'][()])

    def test_process_corpus_with_downsampling(self, processor, tmpdir):
        ds = resources.create_dataset()
        feat_path = os.path.join(tmpdir.strpath, 'feats')

        par_processor = processing.ParallelProcessor(processor)
        par_processor.process_corpus(
            ds,
            feat_path,
            4,
            frame_size=4096,
            hop_size=2048,
            sr=8000
        )

        with h5py.File(feat_path, 'r') as f:
            utts = set(f.keys())

            assert utts == set(ds.utterances.keys())

            assert f['utt-1'].shape == (10, 4096)
            assert f['utt-2'].shape == (10, 4096)
            assert f['utt-3'].shape == (5, 4096)
            assert f['utt-4'].shape == (3, 4096)
            assert f['utt-5'].shape == (10, 4096)

    def test_process_corpus_sets_container_attributes(self, processor, tmpdir):
        ds = resources.create_dataset()
        feat_path = os.path.join(tmpdir.strpath, 'feats')

        par_processor = processing.ParallelProcessor(processor)
        feat_container = par_processor.process_corpus(
            ds,
            feat_path,
            4,
            frame_size=4096,
            hop_size=2048
        )

        with feat_container:
            assert feat_container.frame_size == 4096
            assert feat_container.hop_size == 2048
            assert feat_container.sampling_rate == 16000

    def test_process_corpus_sets_container_attributes_with_downsampling(self, processor, tmpdir):
        ds = resources.create_dataset()
        feat_path = os.path.join(tmpdir.strpath, 'feats')

        par_processor = processing.ParallelProcessor(processor)
        feat_container = par_processor.process_corpus(
            ds,
            feat_path,
            4,
            frame_size=4096,
            hop_size=2048,
            sr=8000
        )

        with feat_container:
            assert feat_container.frame_size == 4096
            assert feat_container.hop_size == 2048
            assert feat_container.sampling_rate == 8000

    def test_process_corpus_with_frame_hop_size_change_stores_correct(self, processor, tmpdir):
        ds = resources.create_dataset()
        feat_path = os.path.join(tmpdir.strpath, 'feats')

        processor.mock_frame_size_scale = 2.5
        processor.mock_hop_size_scale = 5
        par_processor = processing.ParallelProcessor(processor)
        feat_container = par_processor.process_corpus(
            ds,
            feat_path,
            4,
            frame_size=4096,
            hop_size=2048
        )

        with feat_container:
            assert feat_container.frame_size == 10240
            assert feat_container.hop_size == 10240
