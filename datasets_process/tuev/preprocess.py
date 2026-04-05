import argparse
from pathlib import Path
from multiprocessing import Pool

def preprocess(root : Path, config : dict, handle, outdir):
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
    for files, out in zip([train_files, val_files, test_files], [train_outdir, val_outdir, test_outdir]):
        for f in files:
            p = root / f
            folder = p.parent
            fname = p.name

            parameters.append([str(folder), str(fname), out])
            # break
    
    with Pool(processes=4) as pool:
        pool.map(handle, parameters)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--root')
    parser.add_argument('--config')
    parser.add_argument('--model')
    parser.add_argument('--outdir')
    parser.add_argument('--test_only', action='store_true', default=False)

    args = parser.parse_args()

    model : str = args.model

    if model == 'tfm_tokenizer':
        # from .models.tfm import 
        from .models.tfm import load_up_objects
    elif model == 'biot':
        from .models.biot import load_up_objects
    elif model == 'labram':
        from .models.labram import load_up_objects
    else:
        raise NotImplementedError()
    
    import json

    with open(args.config, 'r') as f:
        config = json.load(f)
    
    if args.test_only:
        config['train'] = []
        config['val'] = []

    preprocess(Path(args.root), config, load_up_objects, args.outdir)

if __name__ == '__main__':
    main()