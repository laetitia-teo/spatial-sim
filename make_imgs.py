import os
import pathlib
import re

from multiprocessing import Pool, TimeoutError
from pprint import pprint

from dataset import PartsDataset
from gen import SameConfigGen, CompareConfigGen
from argparse import ArgumentParser

parser = ArgumentParser()

add = parser.add_argument
add('--path', default='data', type=str)
add('--task', default='double', choices=['simple', 'double'])
add('--n_proc', type=int, default=8)
args = parser.parse_args()

def make_imgs(data_files):
    for data_file in data_files:
        file_path = os.path.join(path, data_file)
        generator = SameConfigGen() if task == 'simple' else CompareConfigGen()
        print('Loading file')
        generator.load(file_path)
        print('File loaded')
        print('Dataset created')
        imgpath = os.path.join(args.path, task, 'images', data_file)
        pathlib.Path(imgpath).mkdir(parents=True, exist_ok=True)
        print(f'Rendering data file {data_files}')
        generator.render(imgpath)

task = args.task
path = os.path.join(args.path, task)
data_files = os.listdir(path)
already_processed = os.listdir(os.path.join(args.path, task, 'images'))
data_files = [f for f in data_files if f not in already_processed]
if task == 'double':
    data_files = [f for f in data_files if re.match(r'^rotcur.*$', f) is not None]
else:
    data_files = [f for f in data_files if re.match(r'^[0-9]+_[0-9]+_[0-9]+$', f) is not None]
data_files = [p for p in data_files if not pathlib.Path(os.path.join(path, p)).is_dir()]
print(f'Rendering for task {task}, found {len(data_files)} files')
pprint(data_files)

# divide the data list among processes
data_file_list = []
num_files_per_proc = len(data_files) // args.n_proc
remainder = len(data_files) % args.n_proc
start = 0
for proc_id in range(args.n_proc):
    end = start + num_files_per_proc
    if proc_id < remainder:
        end += 1
    data_file_list.append(data_files[start:end])
    start = end

if __name__ == "__main__":
    # make_imgs(data_files)
    with Pool(processes=args.n_proc) as pool:
        pool.map(make_imgs, data_file_list)
