import pickle
import pathlib

def calculate(path):
    counts = {}
    for file in path.glob('*.pkl'):
        with open(file, 'rb') as f:
            content = pickle.load(f)
            y = content['y']
            counts[y] = counts.get(y, 0) + 1

    print(path, ':\t', counts)

if __name__ == '__main__':
    path = pathlib.Path('./datasets/TUH/tuh_eeg_abnormal/v3.0.1/edf/100/processed/')
    calculate(path / 'train')
    calculate(path / 'val')