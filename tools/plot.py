import matplotlib.pyplot as plt
import numpy as np
csfont = {'fontname': "Times New Roman",
          'size'    : 22}


hist = np.load('entropy_hist.npy', allow_pickle=True)

print(np.shape(hist))
print(np.max(hist))

hist = hist / np.sum(hist)

print(np.max(hist))

xaxis = np.arange(256)

fig, ax = plt.subplots(figsize = (12, 7))
p1 = plt.bar(xaxis, hist, 1)

plt.ylabel('normalized frequency', **csfont)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.savefig('save.png')