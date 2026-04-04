import os
import argparse
from pathlib import Path


from typing import NamedTuple

def ImageNetWoofLoader(base, input_size, mean, std):
    import torch
    from torchvision import transforms as tfs
    from torchvision.transforms.functional import InterpolationMode

    from PIL import Image

    if isinstance(base, str):
        base = Path(base)

    mapping = {
        'n02086240': 155,
        'n02087394': 159,
        'n02088364': 162,
        'n02089973': 167,
        'n02093754': 182,
        'n02096294': 193,
        'n02099601': 207,
        'n02105641': 229,
        'n02111889': 258,
        'n02115641': 273,
    }

    class Item(NamedTuple):
        path : Path
        label : int

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, root_dir : Path):
            self.transform = tfs.Compose([
                tfs.Resize(input_size, interpolation=InterpolationMode.BICUBIC),
                tfs.CenterCrop(input_size),
                tfs.ToTensor(),
                tfs.Lambda(lambda img: img.expand(3, -1, -1)),
                tfs.Normalize(mean, std),
            ])
            # self.transform2 = 

            self.root_dir = root_dir
            self.samples : list[Item] = None
            self.mapping = mapping

        def __list_samples(self):
            samples = []

            for sub in self.root_dir.iterdir():
                for file in sub.glob("*.JPEG"):
                    samples.append(Item(file, mapping[sub.name]))

            self.samples = samples

        def __len__(self):
            if self.samples is None:
                self.__list_samples()
            
            return len(self.samples)

        def __getitem__(self, index):
            if self.samples is None:
                self.__list_samples()

            sample : Item = self.samples[index]
            path, label = sample

            img = Image.open(path)
            img_tensor = self.transform(img)

            return img_tensor, label
        
    train_loader = Dataset(base / 'train')
    val_loader = Dataset(base / 'val')
    test_loader = val_loader

    return train_loader, val_loader, test_loader

def set_seeds(device, args):
    import torch
    import numpy as np

    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)

def prepare_dataloader(args, input_size, mean, std):
    if args.dataset == 'ImageNetWoof':
        # from utils import ImageNetWoofLoader
        train_set, val_set, test_set = ImageNetWoofLoader(args.dataset_dir, input_size, mean, std)
    else:
        raise NotImplementedError()

    from torch.utils.data import DataLoader
    batch_size=  args.batch_size
    num_workers = args.num_workers
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=True,
        drop_last=False
    )
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=True,
        drop_last=False
    )

    return train_loader, val_loader, test_loader

def prepare_model(args) -> 'torch.nn.Module':

    if args.model == 'b16_swag':
        model_name = 'vit_b16_in1k'
        repo = "facebookresearch/swag"
    else:
        raise NotImplementedError()

    import torch
    import patch

    model = torch.hub.load(repo_or_dir=repo, model=model_name)
    model.eval()
    patch.swag_vit(model)

    print(model._tome_info['class_token'])

    r = args.tome_r
    variant = args.tome_variant

    # print(r)
    if len(r) == 0:
        r = 0
        variant = ""
    elif len(r) == 1:
        r : str = r[0]
        if r.startswith("L/"):
            r = [int(r[2:])]
        elif r.startswith("F/"):
            r, f = r[2:].split(',')
            r = (int(r), float(f))
        else:
            r = int(r)
    else:
        r = [int(it) for it in r]

    print(type(r), ":", r)

    model.r = r
    model.variant = variant
    if hasattr(args, 'pivot_factor'):
        model.pivot_factor = args.pivot_factor
    # model._tome_info['pivot_factor'] = args.pivot_factor
    return model

def calculate_metrics(y_hat, y_ground):
    from sklearn.metrics import accuracy_score
    result = {
        "accuracy": accuracy_score(y_ground, y_hat),
        # "threshold": threshold if threshold is not None else 0.5
    }
    return result

def run(args):
    version = f"{args.model}-{args.dataset}-{args.lr}-{args.batch_size}-{args.seed}"
    logdir : str = args.log_dir
    workspace = Path('.', 'workspace', args.workspace)
    log_dir = workspace / 'logs' / logdir / version
    cp_dir = workspace / 'checkpoints' / logdir / version
    prof_dir = workspace / 'profiles' / logdir / version

    # test existance, and fast quit to avoid comflict
    if not args.debug and (log_dir.exists() or cp_dir.exists() or prof_dir.exists()):
        print("workspace {}/{} exists, skipping".format(args.log_dir, version))
        exit()

    print("running {}/{}".format(args.log_dir, version))
    print(args)
    import torch

    device = torch.device('cuda:0')

    # setup seeds
    set_seeds(device, args)

    # define the model
    model = prepare_model(args)
    model = model.to(device)

    # prepare dataloaders
    train_loader, test_loader, val_loader = prepare_dataloader(
        args, 
        384,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    # define optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay,
    )

    def infer_post_fn(pred, label):
        pred = torch.argmax(pred, -1, True)#.flatten(-2, -1)
        return pred, label
    
    from torch.profiler import profile, ProfilerActivity, record_function, _ExperimentalConfig
    from engine import Hooks, create_global_context, train, test, valid

    # from torch.utils.tensorboard import SummaryWriter
    ctx = create_global_context(device, log_dir, cp_dir, slots=1)
    
    hooks = Hooks(
        calc_metric=calculate_metrics,
        handle_post_infer_result=infer_post_fn
    )

    from contextlib import ExitStack
    with ExitStack() as stack:

        if args.profile:
            torch.cuda.memory._record_memory_history()
            perf = stack.enter_context(profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True))

            stack.push(lambda *_: torch.cuda.memory._dump_snapshot(str(prof_dir / "memory.pickle")))
            stack.push(lambda *_: perf.export_chrome_trace(str(prof_dir / 'profile.json')))
        
        if args.valid:
            valid(
                ctx_global=ctx,
                model=model,
                test_dataloader=test_loader,
                hooks=hooks
            )

        if args.test:
            test(
                ctx_global=ctx,
                model=model,
                test_dataloader=test_loader,
                hooks=hooks
                # infer_post_fn=infer_post_fn,
                # metric_fn=calculate_metrics,
            )
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100,
                        help="number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--weight_decay", type=float,
                        default=1e-5, help="weight decay")
    parser.add_argument("--batch_size", type=int,
                        default=32, help="batch size")
    parser.add_argument("--num_workers", type=int,
                        default=4, help="number of workers")
    parser.add_argument("--dataset", type=str, default="ImageNetWoof", help="dataset")
    parser.add_argument(
        "--model", type=str, default="b16_swag", help="which supervised model to use"
    )
    # parser.add_argument(
    #     "--in_channels", type=int, default=16, help="number of input channels"
    # )
    # parser.add_argument(
    #     "--sample_length", type=float, default=10, help="length (s) of sample"
    # )
    # parser.add_argument(
    #     "--n_classes", type=int, default=1, help="number of output classes"
    # )
    # parser.add_argument(
    #     "--sampling_rate", type=int, default=200, help="sampling rate (r)"
    # )
    # parser.add_argument("--token_size", type=int,
    #                     default=200, help="token size (t)")
    # parser.add_argument(
    #     "--hop_length", type=int, default=100, help="token hop length (t - p)"
    # )
    # parser.add_argument(
    #     "--pretrain_model_path", type=str, default="", help="pretrained model path"
    # )

    # modification made for tc_eeg
    parser.add_argument("--seed", type=int, default=1)
    # parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--log_dir", type=str, default='test')
    # parser.add_argument("--workspace", type=str, required=True)
    parser.add_argument("--workspace", type=str, default='test')
    parser.add_argument("--dataset_dir", type=str, default="./models/ToMeViT/datasets/imagewoof2/")
    parser.add_argument("--subset", type=str, default='')

    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--tome_r", type=str, nargs='+', default=[])
    parser.add_argument("--tome_variant", type=str, nargs='+', default=[])
    parser.add_argument("--pivot_factor", type=float, default=None)
    parser.add_argument("--linear", action='store_true', default=False)
    parser.add_argument("--flash", action='store_true', default=False)

    # parser.add_argument("--load_from_checkpoint", type=str, default=None)
    parser.add_argument("--train", action='store_true', default=False)
    parser.add_argument("--test", action='store_true', default=False)
    parser.add_argument("--valid", action='store_true', default=False)
    parser.add_argument("--profile", action='store_true', default=False)
    # parser.add_argument("--rtl_tome", action='store_true', default=False)
    parser.add_argument("--cls_token", action='store_true', default=False)
    parser.add_argument("--debug", action='store_true', default=False)
    # end of modification

    args = parser.parse_args()
    if args.debug:
        args.workspace = 'debug'

    run(args)
