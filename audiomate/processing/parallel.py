import sys
import multiprocessing as mp

import audiomate
from audiomate import containers


class ParallelCorpusProcessor:

    def __init__(self, processor):
        self.processor = processor

    def run(self, corpus,
            output_path,
            num_workers,
            frame_size=400,
            hop_size=160,
            sr=None):

        if num_workers < 2:
            raise ValueError('Parallel processing needs at least 2 workers')

        # ctx = mp.get_context('spawn')

        utterance_ids = list(corpus.utterances.keys())
        num_utts = len(utterance_ids)
        utts_per_worker = int(num_utts / num_workers)

        workers = []
        result_queue = mp.Queue()

        # Setup processes
        # Assign each process a range of utterances
        # Start processes
        for i in range(num_workers):
            utt_start_index = i * utts_per_worker
            utt_end_index = (i + 1) * utts_per_worker

            if i == num_workers - 1:
                utt_end_index = num_utts

            # processor = copy.deepcopy(self.processor)
            print('worker {}: utts {}-{}'.format(i, utt_start_index, utt_end_index))

            worker_utts = utterance_ids[utt_start_index:utt_end_index]

            worker = mp.Process(target=_process_utterances_task,
                                args=(self.processor,
                                      result_queue,
                                      worker_utts,
                                      corpus,
                                      frame_size,
                                      hop_size,
                                      sr
                                      ))

            workers.append(worker)
            worker.start()

        # Retrieve results from processes
        # Write to output feature-container
        feat_container = containers.FeatureContainer(output_path)
        feat_container.open()

        for i in range(num_utts):
            utt_idx, feats = result_queue.get()
            feat_container.set(utt_idx, feats)

        sampling_rate = sr

        if sampling_rate is None:
            sampling_rate = corpus.utterances[utterance_ids[0]].sampling_rate

        tf_frame_size, tf_hop_size = self.processor.frame_transform(frame_size, hop_size)
        feat_container.frame_size = tf_frame_size
        feat_container.hop_size = tf_hop_size
        feat_container.sampling_rate = sr or sampling_rate

        feat_container.close()

        # Wait for processes to finish
        for worker in workers:
            worker.join()

        return feat_container


def _process_utterances_task(processor,
                             result_queue,
                             utterances,
                             corpus,
                             frame_size,
                             hop_size,
                             sr):
    """
    Method executed on every process in parallel corpus processing.
    """
    sampling_rate = -1

    try:
        for utterance_idx in utterances:
            utterance = corpus.utterances[utterance_idx]
            utt_sampling_rate = utterance.sampling_rate

            if sr is None:
                if sampling_rate > 0 and sampling_rate != utt_sampling_rate:
                    raise ValueError(
                        ('File {} has a different sampling-rate'
                         'than the previous ones!').format(utterance.file.idx)
                    )

                sampling_rate = utt_sampling_rate

            data = processor.process_utterance(utterance,
                                               frame_size=frame_size,
                                               hop_size=hop_size,
                                               sr=sr,
                                               corpus=corpus)
            result_queue.put((utterance.idx, data))

        result_queue.close()
    except:
        print(sys.exc_info()[0])
