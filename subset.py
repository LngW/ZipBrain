from pathlib import Path
import numpy as np

def main():

    subset_size = 200

    datasets = Path.home() / 'datasets' / 'TUH' / 'tuh_eeg_abnormal' / 'v3.0.1' / 'edf'
    train_ds = datasets / 'train'
    val_ds = datasets / 'eval'

    chosen = {k : {} for k in ['train', 'val', 'test']}

    for it in ['normal', 'abnormal']:
        files = list((train_ds / it / '01_tcp_ar').glob('*.edf'))
        size = len(files)
        if size > subset_size * 2:
            s = subset_size
        else:
            s = size // 2

        np.random.shuffle(files)

        chosen['train'][it] = files[:s]
        chosen['val'][it] = files[s:s * 2]

        files = list((val_ds / it / '01_tcp_ar').glob('*.edf'))
        size = len(files)
        s = min(size, subset_size)
        np.random.shuffle(files)

        chosen['test'][it] = files[:s]

    dump_folder = Path('.', 'datasets', 'TUH', 'tuh_eeg_abnormal' , 'v3.0.1' , 'edf', str(subset_size), 'original')
    for sub0, data_group in chosen.items():
        for sub1, data in data_group.items():
            to_folder = dump_folder / sub0 / sub1
            to_folder.mkdir(parents=True, exist_ok=False)
            for d in data:
                assert isinstance(d, Path)
                to_file : Path = to_folder / d.name
                to_file.hardlink_to(d)

if __name__ == '__main__':
    main()