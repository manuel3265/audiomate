import os
import tempfile

import pingu
from pingu.corpus import assets


def dummy_wav_path_and_name():
    name = 'test.wav'
    return os.path.join(os.path.dirname(__file__), name), name


def sample_default_ds_path():
    return os.path.join(os.path.dirname(__file__), 'default_ds')


def sample_broadcast_ds_path():
    return os.path.join(os.path.dirname(__file__), 'broadcast_ds')


def sample_kaldi_ds_path():
    return os.path.join(os.path.dirname(__file__), 'kaldi_ds')


def sample_musan_ds_path():
    return os.path.join(os.path.dirname(__file__), 'musan_ds')


def get_wav_file_path(name):
    return os.path.join(os.path.dirname(__file__), 'wav_files', name)


def create_dataset():
    temp_path = tempfile.mkdtemp()

    ds = pingu.Corpus(temp_path)

    wav_1_path = get_wav_file_path('wav_1.wav')
    wav_2_path = get_wav_file_path('wav_2.wav')
    wav_3_path = get_wav_file_path('wav_3.wav')
    wav_4_path = get_wav_file_path('wav_4.wav')

    file_1 = ds.new_file(wav_1_path, file_idx='wav-1')
    file_2 = ds.new_file(wav_2_path, file_idx='wav_2')
    file_3 = ds.new_file(wav_3_path, file_idx='wav_3')
    file_4 = ds.new_file(wav_4_path, file_idx='wav_4')

    issuer_1 = ds.new_issuer('spk-1')
    issuer_2 = ds.new_issuer('spk-2')
    issuer_3 = ds.new_issuer('spk-3')

    utt_1 = ds.new_utterance('utt-1', file_1.idx, issuer_idx=issuer_1.idx)
    utt_2 = ds.new_utterance('utt-2', file_2.idx, issuer_idx=issuer_1.idx)
    utt_3 = ds.new_utterance('utt-3', file_3.idx, issuer_idx=issuer_2.idx, start=0, end=15)
    utt_4 = ds.new_utterance('utt-4', file_3.idx, issuer_idx=issuer_2.idx, start=15, end=25)
    utt_5 = ds.new_utterance('utt-5', file_4.idx, issuer_idx=issuer_3.idx)

    ds.new_label_list(utt_1.idx, labels=assets.Label('who am i'))
    ds.new_label_list(utt_2.idx, labels=assets.Label('who are you'))
    ds.new_label_list(utt_3.idx, labels=assets.Label('who is he'))
    ds.new_label_list(utt_4.idx, labels=assets.Label('who are they'))
    ds.new_label_list(utt_5.idx, labels=assets.Label('who is she'))

    return ds


def create_multi_label_corpus():
    ds = pingu.Corpus()

    wav_1_path = get_wav_file_path('wav_1.wav')
    wav_2_path = get_wav_file_path('wav_2.wav')
    wav_3_path = get_wav_file_path('wav_3.wav')
    wav_4_path = get_wav_file_path('wav_4.wav')

    file_1 = ds.new_file(wav_1_path, file_idx='wav-1')
    file_2 = ds.new_file(wav_2_path, file_idx='wav_2')
    file_3 = ds.new_file(wav_3_path, file_idx='wav_3')
    file_4 = ds.new_file(wav_4_path, file_idx='wav_4')

    issuer_1 = ds.new_issuer('spk-1')
    issuer_2 = ds.new_issuer('spk-2')
    issuer_3 = ds.new_issuer('spk-3')

    utt_1 = ds.new_utterance('utt-1', file_1.idx, issuer_idx=issuer_1.idx)
    utt_2 = ds.new_utterance('utt-2', file_2.idx, issuer_idx=issuer_1.idx)
    utt_3 = ds.new_utterance('utt-3', file_3.idx, issuer_idx=issuer_2.idx, start=0, end=15)
    utt_4 = ds.new_utterance('utt-4', file_3.idx, issuer_idx=issuer_2.idx, start=15, end=25)
    utt_5 = ds.new_utterance('utt-5', file_3.idx, issuer_idx=issuer_2.idx, start=25, end=40)
    utt_6 = ds.new_utterance('utt-6', file_4.idx, issuer_idx=issuer_3.idx, start=0, end=15)
    utt_7 = ds.new_utterance('utt-7', file_4.idx, issuer_idx=issuer_3.idx, start=15, end=25)
    utt_8 = ds.new_utterance('utt-8', file_4.idx, issuer_idx=issuer_3.idx, start=25, end=40)

    ds.new_label_list(utt_1.idx, labels=[
        assets.Label('music', 0, 5),
        assets.Label('speech', 5, 12),
        assets.Label('music', 13, 15)
    ])

    ds.new_label_list(utt_2.idx, labels=[
        assets.Label('music', 0, 5),
        assets.Label('speech', 5, 12),
        assets.Label('music', 13, 15)
    ])

    ds.new_label_list(utt_3.idx, labels=[
        assets.Label('music', 0, 1),
        assets.Label('speech', 2, 6)
    ])

    ds.new_label_list(utt_4.idx, labels=[
        assets.Label('music', 0, 5),
        assets.Label('speech', 5, 12),
        assets.Label('music', 13, 15)
    ])

    ds.new_label_list(utt_5.idx, labels=[
        assets.Label('speech', 0, 7)
    ])

    ds.new_label_list(utt_6.idx, labels=[
        assets.Label('music', 0, 5),
        assets.Label('speech', 5, 12),
        assets.Label('music', 13, 15)
    ])

    ds.new_label_list(utt_7.idx, labels=[
        assets.Label('music', 0, 5),
        assets.Label('speech', 5, 11)
    ])

    ds.new_label_list(utt_8.idx, labels=[
        assets.Label('music', 0, 10)
    ])

    return ds