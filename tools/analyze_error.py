import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

errors = np.load('./work_dirs/mask_vit16_sh/error.npy', allow_pickle=True)
errors_dict = errors.tolist()

i = []
e = []

img_ids = open('./data/cityscapes/train.txt').read().splitlines()

for idx, img_id in tqdm(enumerate(img_ids)):

    a, b = errors_dict.get(img_id)

    # print(img_id)
    i.append(a)
    e.append(b)
    
    # print(errors_dict.get(img_id))


plt.scatter(i,e)
plt.xlabel('ignore')
plt.ylabel('error')
plt.savefig('./relation.png')

