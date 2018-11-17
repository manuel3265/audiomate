import numpy as np
from audiomate import containers


single = containers.FeatureContainer('single.hdf5')
par = containers.FeatureContainer('par.hdf5')

single.open()
par.open()

assert sorted(single.keys()) == sorted(par.keys())

for key in single.keys():
    assert np.allclose(single.get(key), par.get(key))

print('HUI')
