import numpy as np
import audiomate

from audiomate.corpus import subset
from audiomate.processing import pipeline
from audiomate.processing import parallel

from timeit import default_timer as timer


def run_single(corpus, processor, out_path):
    processor.process_corpus(corpus, out_path, frame_size=400, hop_size=160)


def run_par(corpus, processor, out_path):
    parproc = parallel.ParallelCorpusProcessor(processor)
    parproc.run(corpus, out_path, 4, frame_size=400, hop_size=160)


if __name__ == '__main__':
    corpus = audiomate.Corpus.load('/Users/matthi/ZHAW/kws/kws-data/public/TIMIT')

    subsetter = subset.SubsetGenerator(corpus, random_seed=230)
    corpus = subsetter.random_subset(1.0)

    mel = pipeline.MelSpectrogram(n_mels=40)
    mel_db = pipeline.PowerToDb(ref=np.max, parent=mel)
    context = pipeline.AddContext(left_frames=4, right_frames=4, parent=mel_db)

    start = timer()
    # run_single(corpus, context, 'single.hdf5')
    run_par(corpus, mel_db, 'par.hdf5')
    print(timer() - start)
