import glob, pandas as pd

files = pd.DataFrame(glob.glob('JPEGImages/*.jpg'))

files[0] = files[0].apply(lambda r: r.split('/')[-1][:-4])
files = files.sample(frac=1)

num_train = int(len(files)*0.8)
files[:num_train].to_csv('ImageSets/Main/train.txt', header=None, index=False)
files[num_train+1:].to_csv('ImageSets/Main/val.txt', header=None, index=False)
