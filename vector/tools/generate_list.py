import os
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str)    
parser.add_argument('--path', type=str)                                                
args = parser.parse_args()

folders = os.listdir(args.path)

if not os.path.exists("vector/list"):
    os.makedirs("vector/list")

fl = open('vector/list/' + args.name + '_list.txt', 'w')
fn = open('vector/list/' + args.name + '_name.txt', 'w')

for i, folder in enumerate(tqdm(folders, total=len(folders))):

    fn.write(str(i) + ' ' + folder + '\n')

    folder_path = os.path.join(args.path, folder)
    files = os.listdir(folder_path)

    for file in files:
        path = os.path.join(folder_path, file)
        if len(os.listdir(path)) > 0:
            fl.write('{} {}\n'.format(os.path.join(folder_path, file), i))

fl.close()
fn.close()
