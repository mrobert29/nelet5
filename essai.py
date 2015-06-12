import matplotlib.pyplot as plt
import cPickle

titre='test kerario-1000-p'
f = file(titre, 'rb')
params = cPickle.load(f)
f.close()

print params