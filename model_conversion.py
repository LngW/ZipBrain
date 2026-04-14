import torch
from types import SimpleNamespace
from run_infer_multiclass import prepare_model, prepare_dataloader
from engine import Hooks
from einops import rearrange

args = SimpleNamespace(
    model = 'BIOT',
    dataset = 'TUEV',
    seed = 0,
    batch_size = 1,
    num_workers = 0,
    sampling_rate = 200,
)

model = prepare_model(args)
model.variant = 'kiddp[q]'
model.r = 23
model.pivot_factor = 0.5
model.imp_factor = None
# dummy_input = torch.rand((1, 16, 1000))

model.cuda().eval()
hooks = Hooks(None)
_, loader, _ = prepare_dataloader(args, hooks)
x, y = next(iter(loader))

print(x.shape)

# x_ = rearrange(x.float(), 'B N (A T) -> B N A T', T=200) / 100
print(model(x.cuda()))
# print(model(x.cuda()))

torch.onnx.export(
    model, x.cuda(), "model.onnx", opset_version=18, input_names=['x'], output_names=['output'],
    dynamo=True, optimize=True, external_data=False
)