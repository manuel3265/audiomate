import os

import numpy as np
import librosa
import h5py

import pytest

from audiomate import tracks
from audiomate import containers
from audiomate.corpus import assets
from audiomate import processing
from audiomate.processing import parallel

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


@pytest.fixture()
def sample_utterance():
    file_track = tracks.FileTrack('test_file', resources.sample_wav_file('wav_1.wav'))
    utterance = assets.Utterance('test', file_track)
    return utterance


class TestProcessor:

    #
    #   process_corpus
    #

    def test_process_corpus(self, processor, tmpdir):
        ds = resources.create_dataset()
        feat_path = os.path.join(tmpdir.strpath, 'feats')

        paraproc = parallel.ParallelCorpusProcessor(processor)
        paraproc.run(ds, feat_path, 4, frame_size=4096, hop_size=2048)

        with h5py.File(feat_path, 'r') as f:
            utts = set(f.keys())

            assert utts == set(ds.utterances.keys())

            assert f['utt-1'].shape == (20, 4096)
            assert f['utt-2'].shape == (20, 4096)
            assert f['utt-3'].shape == (11, 4096)
            assert f['utt-4'].shape == (7, 4096)
            assert f['utt-5'].shape == (20, 4096)
