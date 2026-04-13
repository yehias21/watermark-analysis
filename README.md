# Watermark Analysis

This repository contains code for analyzing and attacking invisible image
watermarking schemes. It accompanies our **first-place solution** to the
**NeurIPS 2024 *Erasing the Invisible* Stress-Test Challenge for Image
Watermarks** (workshop on GenAI Watermarking, ICLR 2025). The paper is
included in [`assets/60_First_Place_Solution_to_Neu.pdf`](assets/60_First_Place_Solution_to_Neu.pdf).

The project implements watermark encoders / decoders (RivaGAN, StegaStamp,
DWT-DCT, DWT-DCT-SVD, Tree-Ring), a suite of removal attacks (distortion,
regeneration via learned image compressors and diffusion refiners,
embedding-space PGD, and an adaptive VAE-based attack fine-tuned against a
specific watermark), and quality / performance metrics to evaluate the
trade-off between watermark removal and perceptual fidelity.

## Highlights

Our pipeline secured first place in both the **black-box** and **beige-box**
tracks of the challenge.

![Final leaderboard for the black-box and beige-box tracks.](assets/leaderboard.png)

*Final leaderboard: Team-MBZUAI ranks first in both tracks of the NeurIPS 2024
Erasing the Invisible challenge.*

![Side-by-side of original watermarked images (top) and our attacked outputs (bottom).](assets/attack_comparison.png)

*Qualitative results: top row shows original watermarked images, bottom row
shows images after our removal attack. Perceptual fidelity is preserved while
the watermark is erased.*

## Method Overview

### Beige-box track: adaptive VAE attack

For the beige-box setting (watermarking algorithm known, key unknown), we
generate paired (watermarked, inverse-watermarked) data from Stable Diffusion
2.1 with publicly available prompts, then fine-tune the SDXL-Refiner VAE to
map a watermarked input to its inverse-watermarked counterpart under an MSE
loss. This is followed by per-image test-time optimization (LPIPS + SSIM
losses) and CIELAB color/contrast transfer to restore visual quality.

![Dataset-generation and VAE-finetuning pipeline for StegaStamp removal.](assets/stegastamp_pipeline.png)

*Overview of the dataset-generation (paired watermarked / inverse-watermarked
images) and VAE fine-tuning stages of the beige-box attack.*

### Black-box track: cluster-specific diffusion regeneration

For the black-box setting, watermarked images are clustered by spatial and
frequency-domain artifacts, and a cluster-specific removal strategy is
applied. The main component is an image-to-image diffusion attack using the
Stable Diffusion Refiner with ChatGPT-generated semantic captions.

![Image-to-image diffusion pipeline with semantic prompts from ChatGPT.](assets/img2img_pipeline.png)

*Diffusion-based regeneration pipeline: the watermarked image is encoded,
forward-diffused, and denoised by the UNet conditioned on a ChatGPT-generated
caption describing the original content.*

![Four clusters discovered in black-box watermarked images.](assets/blackbox_clusters.png)

*Cluster analysis for the black-box track: images fall into four groups
based on spatial artifacts and Fourier-magnitude signatures (no artifacts,
boundary artifacts, circular patterns, square patterns). Each cluster
receives a tailored attack configuration.*

## Repository Structure

```
watermark-analysis/
|- apply_new_adaptive_attack.py      # Apply a trained adaptive-VAE attack to a folder of images
|- train_new_adaptive_attack.py      # Fine-tune the SDXL-Refiner VAE against a specific watermark
|- create_test_dataset.py            # Generate a watermarked test set with SD + a chosen watermark
|- create_train_data_test.py         # Generate (no-wm, wm, inverse-wm) triplets for attack training
|- prompts_generator.py              # COCO-caption iterator used as Stable Diffusion prompts
|- calculate_performance_metrics.py  # Bit-error rate, AUC-ROC, significance thresholds for decoders
|- calculate_quailty_metrics.py      # LPIPS / MSE / PSNR / SSIM / NMI / FID between two folders
|- trw_perormance_metrics.py         # Tree-Ring performance helpers
|- inspect_onnx_operations.py        # Utility to inspect bundled ONNX watermark models
|- parallel_performance.sh           # Helper for running metric jobs in parallel
|- parallel_quality.sh
|
|- attacks/
|   |- adv.py          # Embedding models (CLIP / ResNet-18 / KL-VAE) used by the PGD attack
|   |- adversial.py    # Warm-up PGD attack in embedding space
|   |- distortion.py   # Classical distortions: rotation, crop, erase, brightness, noise, JPEG, ...
|   |- regeneration.py # Compressive-VAE (CompressAI) and diffusion-refiner regeneration attacks
|
|- watermarks/
|   |- Rivagan.py                               # RivaGAN encoder/decoder (ONNX)
|   |- StegaStamp.py                            # StegaStamp encoder/decoder (ONNX)
|   |- DwtDct.py                                # DWT-DCT and DWT-DCT-SVD watermarks
|   |- Trw.py                                   # Tree-Ring watermark key
|   |- TrwStableDiffusion.py                    # Tree-Ring + Stable Diffusion wrapper
|   |- ModifiedStableDiffusionPipeline.py       # SD pipeline with DDIM inversion support
|   |- PostProccessingWatermarksStableDiffusion.py  # SD wrapper for post-hoc watermarking schemes
|
|- metrics/
|   |- performance/evasion_rate.py   # BER, complex-L1, detection ROC/AUC helpers
|   |- quality/                      # Image / perceptual / distributional quality metrics
|
|- utils/generate_dataset.py
|- notebooks/                        # Exploratory notebooks
|- assets/                           # Paper PDF and figures used in this README
|- coco.json                         # COCO captions used as generation prompts
```

## Installation

The code targets Python 3.10+ and CUDA-enabled PyTorch. Core dependencies:

```
torch torchvision diffusers transformers
onnxruntime-gpu pywavelets opencv-python
scikit-image scikit-learn scipy lpips
pytorch-fid compressai
pandas tqdm matplotlib wandb pillow
```

The RivaGAN and StegaStamp watermark encoders rely on ONNX model files
placed under `watermarks/` (`rivagan_encoder.onnx`, `rivagan_decoder.onnx`,
`stega_stamp.onnx`).

## Usage

### 1. Generate a watermarked test dataset

```bash
python create_test_dataset.py
```

By default this creates 1024 watermarked images (batch size 16) under
`cache/test_dataset_<algorithm>/` with a `messages.csv` mapping each image
to its embedded bit-string. Edit the call at the bottom of the script to
change the algorithm (`rivagan`, `dwtdct`, `dwtdctsvd`, `stegastamp`, `trw`).

### 2. Generate paired attack-training data

```bash
python create_train_data_test.py
```

Produces triplets `(no_watermark, watermark, inverse_watermark)` under
`cache/attack_dataset_<algorithm>/`, used to train the adaptive VAE attack.

### 3. Train the adaptive VAE attack

```bash
python train_new_adaptive_attack.py
```

Fine-tunes an `AutoencoderKL` (SDXL-Refiner VAE) with an LPIPS + MSE loss
to map watermarked inputs to clean / inverse-watermarked targets. Runs log
to W&B and checkpoints are written under `training_runs/run_<timestamp>/`.

### 4. Apply the trained attack to new images

```bash
python apply_new_adaptive_attack.py \
    --input_folder path/to/watermarked \
    --output_folder path/to/attacked \
    --vae_path training_runs/run_<timestamp>/models/best_model.pth
```

### 5. Other attacks

```bash
# Classical distortions
python attacks/distortion.py --input_dir path/to/watermarked --output_dir path/to/distorted

# Embedding-space PGD (edit in-file paths and call at bottom)
python attacks/adversial.py

# Regeneration via learned compressors / diffusion refiner (edit paths in-file)
python attacks/regeneration.py
```

### 6. Evaluate

```bash
# Watermark-decoding performance (BER, AUC, significance thresholds)
python calculate_performance_metrics.py \
    --images_path path/to/attacked \
    --csv_path cache/test_dataset_<algorithm> \
    --algorithm <rivagan|stegastamp|dwtdct|dwtdctsvd>

# Image-quality metrics (LPIPS / MSE / PSNR / SSIM / NMI / FID)
python calculate_quailty_metrics.py \
    --ref_folder path/to/watermarked \
    --target_folder path/to/attacked
```

Parallel wrappers are provided in `parallel_performance.sh` and
`parallel_quality.sh`.

## Citation

If you use this code or the attack methodology, please cite:

```bibtex
@inproceedings{shamshad2025erasing,
  title     = {First-Place Solution to NeurIPS 2024 Invisible Watermark Removal Challenge},
  author    = {Shamshad, Fahad and Bakr, Tameem and Shaaban, Yahia and
               Hussein, Noor and Nandakumar, Karthik and Lukas, Nils},
  booktitle = {1st Workshop on GenAI Watermarking, collocated with ICLR 2025},
  year      = {2025}
}
```

Paper: [`assets/60_First_Place_Solution_to_Neu.pdf`](assets/60_First_Place_Solution_to_Neu.pdf).
