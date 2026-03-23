import json
import argparse
import random
from pathlib import Path

def main(root : Path, seed, out):
    root = root / 'edf'
    ch = '01_tcp_ar'
    train_root = root / 'train'
    test_root = root / 'eval'

    train_normal_root = train_root / 'normal' / ch
    train_abnormal_root = train_root / 'abnormal' / ch
    test_normal_root = test_root / 'normal' / ch
    test_abnormal_root = test_root / 'abnormal' / ch

    train_normal_files = list(train_normal_root.glob('*.edf'))
    train_abnormal_files = list(train_abnormal_root.glob('*.edf'))
    test_normal_files = list(test_normal_root.glob('*.edf'))
    test_abnormal_files = list(test_abnormal_root.glob('*.edf'))


    rnd = random.Random(seed)
    rnd.shuffle(train_normal_files)
    rnd.shuffle(train_abnormal_files)

    length = len(train_normal_files)
    val_normal_files = train_normal_files[length * 8 // 10:]
    train_normal_files = train_normal_files[:length * 8 // 10]

    length = len(train_abnormal_files)
    val_abnormal_files = train_abnormal_files[length * 8 // 10:]
    train_abnormal_files = train_abnormal_files[:length * 8 // 10]

    print('trian, eval, test')
    print('normal: ', len(train_normal_files), len(val_normal_files), len(test_normal_files))
    print('abnormal: ', len(train_abnormal_files), len(val_abnormal_files), len(test_abnormal_files))

    records = {
        'train': [],
        'val': [],
        'test': []
    }

    for idxs, files in enumerate([train_normal_files, train_abnormal_files, val_normal_files, val_abnormal_files, test_normal_files, test_abnormal_files]):
        subset = ['train', 'val', 'test'][idxs // 2]
        label = idxs % 2
        for f in files:
            f_name = str(f.absolute())
            records[subset].append({'file': f_name, 'label': label})

    with open(out, 'w') as handle:
        json.dump(records, handle)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('root', default=None)
    parser.add_argument('seed', default=None)
    parser.add_argument('out', default=None)

    args = parser.parse_args()

    print(args)

    main(Path(args.root), args.seed, Path(args.out))