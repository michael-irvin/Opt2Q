# MW Irvin -- Lopez Lab -- 2019-02-18

"""
To use this:
First, *manually* set `noise_model_sample_size` to some integer, in the western_blot_likelihood_fn python module.
Next, run this file.
"""
import numpy as np
from opt2q.examples.western_blot_example.western_blot_likelihood_fn import likelihood_fn, noise_model_sample_size


calibrated_parameters = [3.75961027, -0.07102133,  0.12315658, -6.71231504,
                         -7.59544621, 0.36464246,  0.28572696,  0.46962568]

l = [likelihood_fn(calibrated_parameters) for x in range(50)]

print("Noise Model Sample Size = ", noise_model_sample_size)
print("Variability = ", str(np.std(l)))