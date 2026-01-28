import pathlib
import numpy as np
import os
import pickle
import mne

def create_minimal_dataset(num_rec = 100, src_path = None, tgt_path = None):

    twenty = num_rec // 5

    for it in ['normal', 'abnormal']:
        path = pathlib.Path(src_path).expanduser() / 'train' / it
        files = list(path.rglob("*.edf"))

        idx = np.random.choice(len(files), num_rec, False)
        choosen : list[pathlib.Path] = [files[it] for it in idx]

        parti_train = choosen[:-twenty]
        parti_valid = choosen[-twenty:]

        for files, folder in [(parti_train, 'train'), (parti_valid, 'val')]:
            for file in files:
                hdlink = pathlib.Path(tgt_path) / folder / it
                hdlink.mkdir(parents=True, exist_ok=True)
                hdlink = hdlink / file.name
                hdlink.hardlink_to(file)

        path = pathlib.Path(src_path).expanduser() / 'eval' / it
        files = list(path.rglob("*.edf"))
        idx = np.random.choice(len(files), twenty, False)
        choosen : list[pathlib.Path] = [files[it] for it in idx]

        for file in choosen:
            hdlink = pathlib.Path(tgt_path) / 'test' / it
            hdlink.mkdir(parents=True, exist_ok=True)
            hdlink = hdlink / file.name
            hdlink.hardlink_to(file)

def split_and_dump(fetch_folder : pathlib.Path, dump_folder, label):
    # fetch_folder, sub, dump_folder, label = params
    for file in fetch_folder.glob("*.edf"):
        if True:
            print("process", file)
            file_path = file
            raw = mne.io.read_raw_edf(file_path, preload=True)
            raw.resample(200)
            ch_name = raw.ch_names
            raw_data = raw.get_data()
            channeled_data = raw_data.copy()[:16]
            try:
                channeled_data[0] = (
                    raw_data[ch_name.index("EEG FP1-REF")]
                    - raw_data[ch_name.index("EEG F7-REF")]
                )
                channeled_data[1] = (
                    raw_data[ch_name.index("EEG F7-REF")]
                    - raw_data[ch_name.index("EEG T3-REF")]
                )
                channeled_data[2] = (
                    raw_data[ch_name.index("EEG T3-REF")]
                    - raw_data[ch_name.index("EEG T5-REF")]
                )
                channeled_data[3] = (
                    raw_data[ch_name.index("EEG T5-REF")]
                    - raw_data[ch_name.index("EEG O1-REF")]
                )
                channeled_data[4] = (
                    raw_data[ch_name.index("EEG FP2-REF")]
                    - raw_data[ch_name.index("EEG F8-REF")]
                )
                channeled_data[5] = (
                    raw_data[ch_name.index("EEG F8-REF")]
                    - raw_data[ch_name.index("EEG T4-REF")]
                )
                channeled_data[6] = (
                    raw_data[ch_name.index("EEG T4-REF")]
                    - raw_data[ch_name.index("EEG T6-REF")]
                )
                channeled_data[7] = (
                    raw_data[ch_name.index("EEG T6-REF")]
                    - raw_data[ch_name.index("EEG O2-REF")]
                )
                channeled_data[8] = (
                    raw_data[ch_name.index("EEG FP1-REF")]
                    - raw_data[ch_name.index("EEG F3-REF")]
                )
                channeled_data[9] = (
                    raw_data[ch_name.index("EEG F3-REF")]
                    - raw_data[ch_name.index("EEG C3-REF")]
                )
                channeled_data[10] = (
                    raw_data[ch_name.index("EEG C3-REF")]
                    - raw_data[ch_name.index("EEG P3-REF")]
                )
                channeled_data[11] = (
                    raw_data[ch_name.index("EEG P3-REF")]
                    - raw_data[ch_name.index("EEG O1-REF")]
                )
                channeled_data[12] = (
                    raw_data[ch_name.index("EEG FP2-REF")]
                    - raw_data[ch_name.index("EEG F4-REF")]
                )
                channeled_data[13] = (
                    raw_data[ch_name.index("EEG F4-REF")]
                    - raw_data[ch_name.index("EEG C4-REF")]
                )
                channeled_data[14] = (
                    raw_data[ch_name.index("EEG C4-REF")]
                    - raw_data[ch_name.index("EEG P4-REF")]
                )
                channeled_data[15] = (
                    raw_data[ch_name.index("EEG P4-REF")]
                    - raw_data[ch_name.index("EEG O2-REF")]
                )
            except:
                with open("tuab-process-error-files.txt", "a") as f:
                    f.write(file + "\n")
                continue
            for i in range(channeled_data.shape[1] // 2000):
                dump_path = dump_folder / (file.stem + "_" + str(i) + ".pkl")
                dump_path.parent.mkdir(parents = True, exist_ok = True)
                with open(dump_path, 'wb') as handle:
                    pickle.dump(
                        {"X": channeled_data[:, i * 2000 : (i + 1) * 2000], "y": label},
                        handle
                    )


if __name__ == '__main__':

    num_rec = 100

    src_path = '~/Datasets/TUH/tuh_eeg_abnormal/v3.0.1/edf/'
    org_path = './datasets/TUH/tuh_eeg_abnormal/v3.0.1/edf/{}/original/'.format(num_rec)
    tgt_path = './datasets/TUH/tuh_eeg_abnormal/v3.0.1/edf/{}/processed/'.format(num_rec)

    # create_minimal_dataset(num_rec, src_path, org_path)

    for folder in ['train', 'val', 'test']:
        for idx, label in enumerate(['normal', 'abnormal']):
            split_and_dump(
                pathlib.Path(org_path) / folder / label, 
                pathlib.Path(tgt_path) / folder,
                idx
            )