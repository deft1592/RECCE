from glob import glob
from os.path import join
root="./dataset/Celeb-DF-v2"
dd="mp4face"
a=glob(join(root, 'YouTube-real', 'images', '*.png'))
print(len(a))