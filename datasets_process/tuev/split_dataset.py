import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def main(root : Path, seed, out):
    # root = root

    root_train = root / 'edf' / 'train'
    root_test = root / 'edf' / 'eval'

    files_train = list(root_train.rglob('*.edf'))
    files_test = list(root_test.rglob('*.edf'))

    files_test = [str(it.relative_to(root)) for it in files_test]

    def load_all_rec_adjacent_edf(root, files : list[Path]):
        pendding = []
        for edf in files:
            data = np.genfromtxt(edf.with_suffix('.rec'), delimiter=',')
            df = pd.DataFrame(data, columns=['ch', 'start', 'end', 'type'])
            df['file'] = str(edf.relative_to(root))
            df = df.astype({'ch': int, 'start': float, 'end': float, 'type': int})
            pendding.append(df)

        df : pd.DataFrame = pd.concat(pendding, axis=0)
        return df
    
    labels_train = load_all_rec_adjacent_edf(root, files_train)
    # labels_test = load_all_rec_adjacent_edf(files_test)

    rnd_state = np.random.RandomState(seed=seed)
    files_train_ = []
    files_eval_ = []
    eval_sample_counts = [3, 5, 5, 5, 16, 21] # choose 1/10 files in train set as eval set
    for i in range(1, 7):
        tmp = labels_train[labels_train.type == i].file.unique().to_numpy()
        rnd_state.shuffle(tmp)

        files_eval_.extend(tmp[:eval_sample_counts[i - 1]])
        files_train_.extend(tmp[eval_sample_counts[i-1]:])

    files_train_ = np.unique(files_train_).tolist()
    files_eval_ = np.unique(files_eval_).tolist()

    records = {
        'train': [],
        'val': [],
        'test': []
    }

    for files, cate in zip([files_train_, files_eval_, files_test], ['train', 'val', 'test']):
        records[cate].extend(files)

    Path(out).parent.mkdir(parents=True, exist_ok=True)

    with open(out, 'w') as handle:
        json.dump(records, handle)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('root', default=None)
    parser.add_argument('seed', type=int, default=None)
    parser.add_argument('out', default=None)

    args = parser.parse_args()

    print(args)

    main(Path(args.root), args.seed, Path(args.out))