from timeit import default_timer as timer

import numpy as np
import audiomate


from audiomate.processing import pipeline
from audiomate import processing


def ser(corpus, pl):
    pl.process_corpus(corpus,
                      'ser_out.hdf5',
                      frame_size=400,
                      hop_size=160,
                      sr=16000)


def par(corpus, pl):
    pp = processing.ParallelProcessor(pl)

    pp.process_corpus(corpus,
                      'ser_out.hdf5',
                      4,
                      frame_size=400,
                      hop_size=160,
                      sr=16000)


mel_extractor = pipeline.MelSpectrogram(n_mels=60)
power_to_db = pipeline.PowerToDb(ref=np.max, parent=mel_extractor)

corpus = audiomate.Corpus.load('../../ZHAW/kws/kws-data/public/mailabs_de')

start = timer()
par(corpus, power_to_db)
dur = timer() - start

print(dur)
