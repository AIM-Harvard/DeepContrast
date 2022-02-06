import pandas as pd
import os
import numpy as np


data_dir = '/mnt/aertslab/USERS/Zezhong/contrast_detection'
meta_file = 'clinical_meta_data.csv'
df = pd.read_csv(os.path.join(data_dir, meta_file))
IDs = []
for manufacturer, model in zip(df['manufacturer'], df['manufacturermodelname']):
    ID = str(manufacturer) + ' ' + str(model)
    IDs.append(ID)
df['ID'] = IDs
#print(df['manufacturer'].value_counts())
#print(df['manufacturermodelname'].value_counts())
#print(df['ID'].value_counts())
#print(df.shape[0])

## KVP
print('kvp mean:', df['kvp'].mean().round(3))
print('kvp median:', df['kvp'].median())
print('kvp mode:', df['kvp'].mode())
print('kvp std:', df['kvp'].std())
print('kvp min:', df['kvp'].min())
print('kvp max:', df['kvp'].max())

## slice thickness
print('thk mean:', df['slicethickness'].mean().round(3))
print('thk median:', df['slicethickness'].median())
print('thk mode:', df['slicethickness'].mode())
print('thk std:', df['slicethickness'].std().round(3))
print('thk min:', df['slicethickness'].min())
print('thk max:', df['slicethickness'].max())
print(df['slicethickness'].value_counts())
print(df['slicethickness'].shape[0])

## spatial resolution
print(df['rows'].value_counts())

## pixel spacing
pixels = []
for pixel in df['pixelspacing']:
    pixel = pixel.split("'")[1]
    pixel = float(pixel)
    pixels.append(pixel)
df['pixel'] = pixels
df['pixel'].round(3)
print('pixel mean:', df['pixel'].mean().round(3))
print('pixel median:', df['pixel'].median().round(3))
print('pixel mode:', df['pixel'].mode().round(3))
print('pixel std:', df['pixel'].std())
print('pixel min:', df['pixel'].min())
print('pixel max:', df['pixel'].max())



