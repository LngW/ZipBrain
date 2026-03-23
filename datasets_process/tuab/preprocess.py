import argparse
from pathlib import Path
from multiprocessing import Pool

def preprocess(config, handle, outdir):
    train_files = config['train']
    val_files = config['val']
    test_files = config['test']

    outdir = Path(outdir)

    train_outdir = outdir / 'train'
    val_outdir = outdir / 'val'
    test_outdir = outdir / 'test'

    train_outdir.mkdir(parents=True, exist_ok=True)
    val_outdir.mkdir(parents=True, exist_ok=True)
    test_outdir.mkdir(parents=True, exist_ok=True)

    parameters = []
    for files, outdir in zip([train_files, val_files, test_files], [train_outdir, val_outdir, test_outdir]):
        for cfg in files:
            p = Path(cfg['file'])
            folder = p.parent
            name = p.name
            label = cfg['label']

            parameters.append([folder, name, outdir, label])
    
    with Pool(processes=4) as pool:
        pool.map(handle, parameters)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model')
    parser.add_argument('--config')
    parser.add_argument('--outdir')
    parser.add_argument('--test_only', action='store_true', default=False)

    args = parser.parse_args()

    model : str = args.model

    if model == 'tfm_tokenizer':
        from thirdparty.TFM_Tokenizer.datasets_processing.TUAB.process import split_and_dump
    elif model == 'biot':
        from thirdparty.BIOT.datasets.TUAB.process import split_and_dump
    elif model == 'labram':
        from thirdparty.LaBraM.dataset_maker.make_TUAB import split_and_dump
    else:
        raise NotImplementedError()
    
    import json

    with open(args.config, 'r') as f:
        config = json.load(f)
    
    if args.test_only:
        config['train'] = []
        config['val'] = []

    preprocess(config, split_and_dump, args.outdir)

if __name__ == '__main__':
    main()