from pathlib import Path
from typing import NamedTuple

def ImageNetWoofLoader(base : Path | str, input_size, mean, std):
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