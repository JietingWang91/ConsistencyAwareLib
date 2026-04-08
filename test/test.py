# Basic example
import numpy as np

from evaluation_index.adjusted import PureAccuracy, StandardizedAccuracy, SGINI, PHSIC

pa = PureAccuracy([1, 2, 1, 2], [1, 2, 2, 2],2)
sa = StandardizedAccuracy([1, 2, 1, 2], [1, 2, 2, 2],2)
sgi = SGINI([1, 1, 2, 2], [1, 2, 1, 2])
h = PHSIC(np.array([[0.1], [0.2], [0.3], [0.4]],dtype=float), np.array([[1.0], [0.0], [1.0], [0.0]],dtype=float))

print("PureAccuracy:", pa)
print("StandardizedAccuracy:", sa)
print("SGINI:", sgi)
print("PHSIC:", h)
