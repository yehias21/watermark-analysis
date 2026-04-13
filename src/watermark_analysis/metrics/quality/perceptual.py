import torch
from PIL import Image
from torchvision import transforms

from .lpips import LPIPS

try:  # pragma: no cover - optional dep, Watson loss not bundled
    from .watson import LossProvider  # type: ignore
except Exception:  # noqa: BLE001
    LossProvider = None  # type: ignore

_TO_TENSOR = transforms.ToTensor()


def to_tensor(images):
    return torch.stack([_TO_TENSOR(img) for img in images])

# Module-level constants
SUPPORTED_METRICS = ("lpips", "watson")
LPIPS_MODES = ("vgg", "alex")
WATSON_MODES = ("vgg", "dft", "fft")


def load_perceptual_models(metric_name, mode, device=torch.device("cuda")):
    assert metric_name in SUPPORTED_METRICS
    if metric_name == "lpips":
        assert mode in LPIPS_MODES
        perceptual_model = LPIPS(net=mode).to(device)
    elif metric_name == "watson":
        assert mode in WATSON_MODES
        perceptual_model = (
            LossProvider()
            .get_loss_function(
                "Watson-" + mode, colorspace="RGB", pretrained=True, reduction="none"
            )
            .to(device)
        )
    else:
        assert False
    return perceptual_model


# Compute metric between two images
def compute_metric(image1, image2, perceptual_model, device=torch.device("cuda")):
    assert isinstance(image1, Image.Image) and isinstance(image2, Image.Image)
    image1_tensor = to_tensor([image1]).to(device)
    image2_tensor = to_tensor([image2]).to(device)
    return perceptual_model(image1_tensor, image2_tensor).cpu().item()


# Compute LPIPS distance between two images
def compute_lpips(image1, image2, mode="alex", device=torch.device("cuda")):
    perceptual_model = load_perceptual_models("lpips", mode, device)
    return compute_metric(image1, image2, perceptual_model, device)


# Compute Watson distance between two images
def compute_watson(image1, image2, mode="dft", device=torch.device("cuda")):
    perceptual_model = load_perceptual_models("watson", mode, device)
    return compute_metric(image1, image2, perceptual_model, device)


# Compute metrics between pairs of images
def compute_perceptual_metric_repeated(
    images1,
    images2,
    metric_name,
    mode,
    model,
    device,
):
    # Accept list of PIL images
    assert isinstance(images1, list) and isinstance(images1[0], Image.Image)
    assert isinstance(images2, list) and isinstance(images2[0], Image.Image)
    assert len(images1) == len(images2)
    if model is None:
        model = load_perceptual_models(metric_name, mode).to(device)
    return (
        model(to_tensor(images1).to(device), to_tensor(images2).to(device))
        .detach()
        .cpu()
        .numpy()
        .flatten()
        .tolist()
    )


# Compute LPIPS distance between pairs of images
def compute_lpips_repeated(
    images1,
    images2,
    mode="alex",
    model=None,
    device=torch.device("cuda"),
):
    return compute_perceptual_metric_repeated(
        images1, images2, "lpips", mode, model, device
    )


# Compute Watson distance between pairs of images
def compute_watson_repeated(
    images1,
    images2,
    mode="dft",
    model=None,
    device=torch.device("cuda"),
):
    return compute_perceptual_metric_repeated(
        images1, images2, "watson", mode, model, device
    )
