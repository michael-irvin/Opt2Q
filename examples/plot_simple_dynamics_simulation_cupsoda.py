"""
Plot of simulator results
--------------------------
Simple example of the Opt2Q simulator.

"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from opt2q.simulator import Simulator
from pysb.examples.michment import model
from pysb.pathfinder import *

set_path('cupsoda','/home/pinojc/git/cupSODA') 
new_params = pd.DataFrame([[np.nan, 'normal', 1],
                           [10.0,   'slow',   1],
                           [1.0e3,  'fast',   1]],
                          columns=['kcat', 'condition', 'experiment'])
sim = Simulator(model=model, param_values=new_params, solver='cupsoda')
results = sim.run(np.linspace(0, 50, 50))

results_df = results.opt2q_dataframe

# plot
cm = plt.get_cmap('tab10')
fig, ax = plt.subplots(figsize=(8,6))
for i, (label, df) in enumerate(results_df.groupby(['experiment', 'condition'])):
    df.plot.line(y='Product', ax=ax, label=label, color=cm.colors[i])
plt.legend()
plt.savefig('test.png')
plt.show()
