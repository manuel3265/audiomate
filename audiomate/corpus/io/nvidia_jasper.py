import os
import json
import functools
import multiprocessing

from tqdm import tqdm

import audiomate
from . import base
from audiomate import tracks
from audiomate.utils import audio


class NvidiaJasperWriter(base.CorpusWriter):
    """
    Writes files to use for training with NVIDIA Jasper
    (https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechRecognition/Jasper).

    **Audio Files:**
    Since utterances are expected to be in a separate file and have
    a specific format (WAV 16-Bit PCM), any utterances that do not meet those
    requirements, are extracted into a separate file in the
    subfolder `audio` of the target path.

    **Subviews:**
    Subviews in the Jasper format are represented as different
    Json-Files. For every subview in the corpus a separate
    Json-File is created. Additionaly, a ``all.json`` is created,
    that contains all utterances.

    Args:
        export_all_audio (bool): If ``True``, all utterances are exported,
                                 whether they are in a separate file
                                 already or not.
        transcription_label_list_idx (str): The transcriptions are used from
                                            the label-list with this id.
        num_workers (int): Number of processes to use to process utterances.
    """

    def __init__(self, export_all_audio=False,
                 transcription_label_list_idx=audiomate.corpus.LL_WORD_TRANSCRIPT,
                 num_workers=1):
        self.export_all_audio = export_all_audio
        self.transcription_label_list_idx = transcription_label_list_idx
        self.num_workers = num_workers

    @classmethod
    def type(cls):
        return 'nvidia-jasper'

    def _save(self, corpus, path):
        if self.num_workers > 1:
            utts = self.process_utterances_parallel(corpus, path)
        else:
            utts = self.process_utterances(corpus, path)

        print('sort')
        utts = sorted(utts, key=lambda x: x[1]['original_duration'])

        subviews = {}

        subviews['all'] = [u[1] for u in utts]

        print('subviews')
        for subview_name, subview in corpus.subviews.items():
            utt_filter = subview.utterances.keys()
            subview_utts = [u[1] for u in utts if u[0] in utt_filter]
            subviews[subview_name] = subview_utts

        print('write')
        for name, data in subviews.items():
            target_path = os.path.join(path, '{}.json'.format(name))
            with open(target_path, 'w') as f:
                json.dump(data, f)

    def process_utterances(self, corpus, base_path):
        os.makedirs(os.path.join(base_path, 'audio'), exist_ok=True)

        utts = self.process_utterance(corpus.utterances.values(), base_path)
        return utts

    def process_utterances_parallel(self, corpus, base_path):
        os.makedirs(os.path.join(base_path, 'audio'), exist_ok=True)
        in_utts = list(corpus.utterances.values())
        num_utts = len(in_utts)
        utts_per_worker = int(num_utts / self.num_workers)

        splits = []

        for i in range(self.num_workers - 1):
            start = i * utts_per_worker
            end = (i + 1) * utts_per_worker
            splits.append(in_utts[start:end])

        splits.append(in_utts[end:])

        with multiprocessing.Pool(self.num_workers) as p:
            func = functools.partial(
                self.process_utterance,
                base_path=base_path
            )

            split_res = list(p.imap(func, splits))

        utts = []

        for r in split_res:
            utts.extend(r)

        return utts

    def process_utterance(self, utts, base_path):
        res = []

        for utt in utts:
            export_utt = False

            if self.export_all_audio:
                export_utt = True

            elif utt.start != 0 or utt.end != float('inf'):
                export_utt = True

            elif type(utt.track) != tracks.FileTrack:
                export_utt = True

            elif not utt.track.path.endswith('wav'):
                export_utt = True

            elif utt.sampling_rate != 16000:
                export_utt = True

            if export_utt:
                file_path = os.path.join(
                    base_path,
                    'audio',
                    '{}.wav'.format(utt.idx)
                )

                samples = utt.read_samples(16000)
                audio.write_wav(file_path, samples, 16000)

                num_samples = len(samples)
                duration = num_samples / 16000
            else:
                file_path = utt.track.path
                num_samples = utt.num_samples()
                duration = utt.duration

            rel_path = os.path.relpath(file_path, base_path)

            out_dict = {
                'transcript': utt.label_lists[self.transcription_label_list_idx].join(),
                'files': [{
                    'fname': rel_path,
                    'channels': 1,
                    'sample_rate': 16000,
                    'duration': duration,
                    'num_samples': num_samples,
                    'speed': 1
                }],
                'original_duration': duration,
                'original_num_samples': num_samples
            }

            res.append((utt.idx, out_dict))

        return res
