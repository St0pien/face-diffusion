from argparse import ArgumentParser
from data.datasplitter import DatasetSplitter, default_splits

parser = ArgumentParser(prog="Prepare datasets",
                        description="Load images from kaggle split them into datasets")

parser.add_argument(
    '--out-dir', help='location for generated datasets', default='datasets')
parser.add_argument('--seed', help='seed for random splitting', default='None')
parser.add_argument(
    '--n-threads', help='max number of threads to use', default='16')

options = parser.parse_args()

seed = None
try:
    seed = int(options.seed)
except:
    pass

n_threads = 16
print(options)

try:
    n_threads = int(options['n_threads'])
except:
    pass

splitter = DatasetSplitter(default_splits, seed)

splitter.split_dataset(options.out_dir, max_workers=n_threads)
