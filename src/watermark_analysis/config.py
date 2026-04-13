"""Centralised configuration: paths, seeds, default sizes, model names.

Collects constants that were previously scattered as module-level globals
across the legacy scripts, so behaviour stays numerically identical but
magic numbers live in a single place.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch

# --------------------------------------------------------------------------- #
# Paths                                                                        #
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = REPO_ROOT / "assets"
DATA_DIR = REPO_ROOT / "data"
COCO_PATH = DATA_DIR / "coco.json"
CACHE_DIR = REPO_ROOT / "cache"

# ONNX watermark models: default to the historical ``watermarks/`` checkpoint
# location at repo root. Override with the ``WATERMARK_ONNX_DIR`` env var.
WATERMARK_ONNX_DIR = Path(os.environ.get("WATERMARK_ONNX_DIR", REPO_ROOT / "watermarks"))
RIVAGAN_ENCODER_PATH = str(WATERMARK_ONNX_DIR / "rivagan_encoder.onnx")
RIVAGAN_DECODER_PATH = str(WATERMARK_ONNX_DIR / "rivagan_decoder.onnx")
STEGASTAMP_MODEL_PATH = str(WATERMARK_ONNX_DIR / "stega_stamp.onnx")

PERFORMANCE_OUTPUT_DIR = "./performance"
QUALITY_OUTPUT_DIR = "./quality"

# --------------------------------------------------------------------------- #
# Seeds / dataset sizes                                                        #
# --------------------------------------------------------------------------- #
TEST_DATASET_SEED = 99
ATTACK_DATASET_SEED = 69
PROMPTS_SEED = 69
DEFAULT_NUM_SAMPLES = 1024
DEFAULT_BATCH_SIZE = 16
DEFAULT_NUM_WORKERS = 24

# --------------------------------------------------------------------------- #
# Watermark algorithms                                                         #
# --------------------------------------------------------------------------- #
RIVAGAN_DWT_MSG_LEN = 32
STEGASTAMP_MSG_LEN = 100
TRW_MSG_LEN = 16
BIT_ALGORITHMS = ("rivagan", "dwtdct", "dwtdctsvd")

ORT_PROVIDERS = [
    ("CUDAExecutionProvider", {"device_id": 0}),
    "CPUExecutionProvider",
]

# StegaStamp preprocessing
STEGASTAMP_DOWN_SIZE = (400, 400)
STEGASTAMP_UP_SIZE = (512, 512)

# DwtDct
DWTDCT_DEFAULT_SCALES = [0, 36, 36]
DWTDCT_DEFAULT_BLOCK = 4
DWTDCT_MSG_LEN = 32
DWTDCT_THRESHOLD = 127

# --------------------------------------------------------------------------- #
# Stable-diffusion defaults                                                    #
# --------------------------------------------------------------------------- #
SD_MODEL_V2_1 = "stabilityai/stable-diffusion-2-1"
SD_MODEL_V2 = "stabilityai/stable-diffusion-2"
SD_MODEL_V1_5 = "sd-legacy/stable-diffusion-v1-5"
SDXL_REFINER_MODEL = "stabilityai/stable-diffusion-xl-refiner-1.0"

SD_NUM_INFERENCE_STEPS = 20
SD_GUIDANCE_SCALE = 7.5
SD_HEIGHT = 512
SD_WIDTH = 512

# --------------------------------------------------------------------------- #
# TRW                                                                          #
# --------------------------------------------------------------------------- #
TRW_REVERSAL_INFERENCE_STEPS = 20
TRW_CHANNEL = -1
TRW_PATTERN = "ring"
TRW_MASK_SHAPE = "circle"
TRW_INJECTION_TYPE = "complex"
TRW_RADIUS = 10
TRW_IMAGE_SIZE = 64
TRW_DEVICE_OVERRIDE = "cuda:7"

# --------------------------------------------------------------------------- #
# Adaptive VAE attack                                                          #
# --------------------------------------------------------------------------- #
ADAPTIVE_IMAGE_SIZE = (512, 512)
ADAPTIVE_TRAIN_VAL_SPLIT = 0.75
ADAPTIVE_RUNS_DIR = "training_runs"
ADAPTIVE_RUN_SUBDIRS = ("models", "plots", "logs", "samples")
ADAPTIVE_WANDB_PROJECT = "new_adaptive-attack"
ADAPTIVE_MAX_SAMPLES_TO_PLOT = 4

# --------------------------------------------------------------------------- #
# Adversarial PGD attack                                                       #
# --------------------------------------------------------------------------- #
ADV_EPS_FACTOR = 1 / 255
ADV_ALPHA_FACTOR = 0.05
ADV_N_STEPS = 200
ADV_BATCH_SIZE = 256
ADV_NUM_WORKERS = 24
ADV_DEFAULT_EPS = 8 / 255
ADV_DEFAULT_ALPHA = 2 / 255
PNG_EXT = ".png"

OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
_CUDA = torch.cuda.is_available()
RESNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
RESNET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
if _CUDA:
    RESNET_MEAN = RESNET_MEAN.cuda()
    RESNET_STD = RESNET_STD.cuda()
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
CLIP_INPUT_SIZE = (224, 224)
RESNET_INPUT_SIZE = [224, 224]
RESNET_LAYER_MAP = {
    "layer1": -6,
    "layer2": -5,
    "layer3": -4,
    "layer4": -3,
    "last": -1,
}

# --------------------------------------------------------------------------- #
# Distortion attack                                                            #
# --------------------------------------------------------------------------- #
DEFAULT_DISTORTION_STRENGTHS = {
    "rotation": 25,
    "resizedcrop": 0.5,
    "erasing": 0.25,
    "brightness": 0.6,
    "contrast": 0.8,
    "blurring": 7,
    "noise": 0.1,
    "compression": 80,
}

# --------------------------------------------------------------------------- #
# Metrics                                                                      #
# --------------------------------------------------------------------------- #
VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
FID_BATCH_SIZE = 50
FID_DIMS = 2048
METRIC_KEYS = ("LPIPS", "MSE", "PSNR", "SSIM", "NMI")
PERF_DECISION_THRESHOLD = 0.5
PERF_ALPHA_05 = 0.05
PERF_ALPHA_01 = 0.01
