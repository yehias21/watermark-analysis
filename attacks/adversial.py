import os

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
from tqdm import tqdm

from adv import (
    ClipEmbedding,
    ResNet18Embedding,
    VAEEmbedding,
)

# Module-level constants
EPS_FACTOR = 1 / 255
ALPHA_FACTOR = 0.05
N_STEPS = 200
BATCH_SIZE = 256
DEFAULT_NUM_WORKERS = 24
DEFAULT_EPS = 8 / 255
DEFAULT_ALPHA = 2 / 255
PNG_EXT = ".png"


class WarmupPGDEmbedding:
    def __init__(
        self,
        model,
        device,
        eps: float = DEFAULT_EPS,
        alpha: float = DEFAULT_ALPHA,
        steps: int = 10,
        loss_type: str = "l2",
        random_start: bool = True,
    ):
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.loss_type = loss_type
        self.random_start = random_start
        self.device = device

        # Initialize the loss function
        if self.loss_type == "l1":
            self.loss_fn = torch.nn.L1Loss()
        elif self.loss_type == "l2":
            self.loss_fn = torch.nn.MSELoss()
        else:
            raise ValueError("Unsupported loss type")

    def forward(self, images, init_delta=None):
        self.model.eval()
        images = images.clone().detach().to(self.device)

        # Get the original embeddings
        original_embeddings = self.model(images).detach()
        # initialize adv images
        if self.random_start:
            adv_images = images.clone().detach()
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(
                -self.eps, self.eps
            )
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        elif init_delta is not None:
            clamped_delta = torch.clamp(init_delta, min=-self.eps, max=self.eps)
            adv_images = images.clone().detach() + clamped_delta
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()
        else:
            assert False

        # PGD
        for _ in tqdm(range(self.steps)):
            self.model.zero_grad()
            adv_images.requires_grad = True
            adv_embeddings = self.model(adv_images)

            # Calculate loss
            cost = self.loss_fn(adv_embeddings, original_embeddings)

            # Update adversarial images
            grad = torch.autograd.grad(
                cost, adv_images, retain_graph=False, create_graph=False
            )[0]
            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images


def _build_embedding_model(encoder: str):
    if encoder == "resnet18":
        # we use last layer's state as the embedding
        return ResNet18Embedding("last")
    if encoder == "clip":
        return ClipEmbedding()
    if encoder == "klvae16":
        return VAEEmbedding("MudeHui/vae-f16-c16")
    raise ValueError(f"Unsupported encoder: {encoder}")


def adv_emb_attack(
    wm_img_path, encoder, strength, output_path, device=torch.device("cuda")
):
    # check if the file/directory paths exist
    os.makedirs(output_path, exist_ok=True)
    for path in [wm_img_path, output_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"The path does not exist: {path}")

    # load embedding model
    embedding_model = _build_embedding_model(encoder).to(device)
    embedding_model.eval()
    print("Embedding Model loaded!")

    # load data
    transform = transforms.ToTensor()
    wm_dataset = SimpleImageFolder(wm_img_path, transform=transform)
    wm_loader = DataLoader(
        wm_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=DEFAULT_NUM_WORKERS, pin_memory=True
    )
    print("Data loaded!")

    # Create an instance of the attack
    attack = WarmupPGDEmbedding(
        model=embedding_model,
        eps=EPS_FACTOR * strength,
        alpha=ALPHA_FACTOR * EPS_FACTOR * strength,
        steps=N_STEPS,
        device=device,
    )
    # Generate adversarial images
    for i, (images, image_paths) in tqdm(enumerate(wm_loader)):
        images = images.to(device)
        # shape
        print(images.shape)
        print(f"Processing batch {i+1}...")
        # PGD attack
        images_adv = attack.forward(images)
        # save images
        for img_adv, image_path in zip(images_adv, image_paths):
            save_path = os.path.join(output_path, os.path.basename(image_path))
            save_image(img_adv, save_path)
    print("Attack finished!")
    return


class SimpleImageFolder(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform

        # Load filenames from the root
        self.filenames = [
            os.path.join(root, f)
            for f in os.listdir(root)
            if os.path.isfile(os.path.join(root, f))
            and os.path.splitext(f)[1].lower() in PNG_EXT
        ]

    def __getitem__(self, index):
        image_path = self.filenames[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, image_path  # return image path to identify the image file later

    def __len__(self):
        return len(self.filenames)


for attack in ['clip', 'resnet18']:
    for watermark in ['dwtdct']:
        adv_emb_attack(
            f'/ephemeral/tbakr/watermark-analysis/cache/test_dataset_{watermark}',
            attack,
            4,
            f'/ephemeral/tbakr/watermark-analysis/attacked/test_dataset_{watermark}_{attack}',
        )
