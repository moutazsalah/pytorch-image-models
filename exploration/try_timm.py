"""Lab 5, Task 4: two documented timm functionalities, run on timm v1.0.31."""
from io import BytesIO
from urllib.request import urlopen

import torch
from PIL import Image
import timm

print("timm", timm.__version__, "| torch", torch.__version__)
print("models:", len(timm.list_models()), "| with pretrained weights:", len(timm.list_models(pretrained=True)))

# Functionality 1: pretrained inference (hfdocs/source/quickstart.mdx, "Using Pretrained Models for Inference")
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/timm/cat.jpg"
image = Image.open(BytesIO(urlopen(url).read())).convert("RGB")
model = timm.create_model("mobilenetv3_large_100", pretrained=True).eval()
transform = timm.data.create_transform(**timm.data.resolve_model_data_config(model))
x = transform(image).unsqueeze(0)
with torch.no_grad():
    probs = model(x).softmax(dim=-1)[0]
values, indices = torch.topk(probs, 5)
labels = timm.data.ImageNetInfo().index_to_description
print("\n[1] input", tuple(x.shape), "-> top-5 indices", indices.tolist())
for v, i in zip(values, indices):
    print(f"    {i.item():4d}  {labels(i.item()):<30s} {v.item():.4f}")

# Functionality 2: multi-scale feature maps (hfdocs/source/feature_extraction.mdx, "Create a feature map extraction model")
fm = timm.create_model("resnest26d", features_only=True, pretrained=True).eval()
with torch.no_grad():
    feats = fm(torch.randn(2, 3, 224, 224))
print("\n[2] channels", fm.feature_info.channels(), "| reductions", fm.feature_info.reduction())
for f in feats:
    print("    ", tuple(f.shape))
