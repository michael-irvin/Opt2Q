import pickle
import numpy as np
from matplotlib import pyplot as plt


with open(f'synthetic_WB_dataset_1500s_2020_12_3.pkl', 'rb') as data_input:
    dataset = pickle.load(data_input)

cm = plt.get_cmap('tab10')
plt.fig3, ax1 = plt.subplots(figsize=(8, 2))
ax1.set_title('Rendering of an Immunoblot consistent with the \n synthetic ordinal measurements of tBID concentration')
ax1.scatter(x=dataset.data['time'],
            y=np.ones_like(dataset.data['time']), marker="_",
            linewidth=1*(dataset.data['tBID_blot'].values+1), s=400,
            color=cm.colors[1], label=f'tBID ordinal data', alpha=0.7)
plt.show()
