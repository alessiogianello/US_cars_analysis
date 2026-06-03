import numpy as np


ages = [12, 33, 44, 53, 21, 22]
ages_array = np.array(ages)
under_30_mask = ages_array < 30
under_30_vector = ages_array[under_30_mask]
print(under_30_vector)
