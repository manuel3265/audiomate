from audiomate.corpus import io
from audiomate.corpus import subset
import audiomate


# full = audiomate.Corpus.load('/Users/matthi/Downloads/cy/', reader='common-voice')
# print(full.subviews)
# ds = full.subviews['test']

full = audiomate.Corpus.load('/Users/matthi/Downloads/mailabs', reader='mailabs')
g = subset.SubsetGenerator(full)
ds = g.random_subset(0.1)


w = io.NvidiaJasperWriter(
    export_all_audio=True,
    num_workers=1
)

w.save(ds, '/Users/matthi/Downloads/testout')
