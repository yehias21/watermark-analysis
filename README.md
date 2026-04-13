# Watermark Analysis

First-place solution to the **NeurIPS 2024 *Erasing the Invisible* Stress-Test
Challenge for Image Watermarks** (workshop on GenAI Watermarking, ICLR 2025).
Paper: [`assets/60_First_Place_Solution_to_Neu.pdf`](assets/60_First_Place_Solution_to_Neu.pdf).

The repo implements watermark encoders/decoders (RivaGAN, StegaStamp, DWT-DCT,
DWT-DCT-SVD, Tree-Ring), removal attacks (distortion, diffusion regeneration,
embedding-space PGD, adaptive VAE), and quality/performance metrics.

![Final leaderboard.](assets/leaderboard.png)

![Originals (top) vs. attacked outputs (bottom).](assets/attack_comparison.png)

## Method

**Beige-box (algorithm known, key unknown).** Generate paired
(watermarked, inverse-watermarked) data from SD 2.1, fine-tune the SDXL-Refiner
VAE to invert the watermark under MSE, then per-image LPIPS+SSIM optimization
and CIELAB color transfer.

![Adaptive VAE pipeline.](assets/stegastamp_pipeline.png)

**Black-box.** Cluster images by spatial/frequency artifacts and apply a
cluster-specific image-to-image diffusion attack (SD Refiner) with
ChatGPT-generated captions.

![Diffusion regeneration pipeline.](assets/img2img_pipeline.png)

## Layout

```
scripts/                      # CLI entry points
src/watermark_analysis/
    config.py io.py prompts.py datasets.py cli.py
    watermarks/               # encoders/decoders + ABC + registry
    attacks/                  # distortion, regeneration, adversarial, adaptive/
    metrics/                  # performance + quality (image, perceptual, FID)
assets/  data/  notebooks/
```

## Install

```bash
pip install -e .
```

ONNX weights (`rivagan_encoder.onnx`, `rivagan_decoder.onnx`,
`stega_stamp.onnx`) go under `watermarks/` or set `WATERMARK_ONNX_DIR`.

## Usage

```bash
# 1. Watermarked test set
python scripts/create_dataset.py --split test --watermark dwtdctsvd

# 2. Paired training data for the adaptive attack
python scripts/create_dataset.py --split attack --watermark dwtdct

# 3. Train the VAE attack
python scripts/train_adaptive_attack.py \
    --cache-dir cache/attack_dataset_dwtdct --mode no_watermark

# 4. Apply
python scripts/apply_adaptive_attack.py \
    --input-folder path/to/watermarked \
    --output-folder path/to/attacked \
    --vae-path training_runs/run_<ts>/models/best_model.pth

# 5. Evaluate
python scripts/eval_performance.py --images_path ... --csv_path ... --algorithm dwtdct
python scripts/eval_quality.py --ref_folder ... --target_folder ...
```

Other attacks: `from watermark_analysis.attacks import DistortionAttacks, VAEAttack, DiffuserAttack, adv_emb_attack`.

## Citation

```bibtex
@inproceedings{shamshad2025erasing,
  title     = {First-Place Solution to NeurIPS 2024 Invisible Watermark Removal Challenge},
  author    = {Shamshad, Fahad and Bakr, Tameem and Shaaban, Yahia and
               Hussein, Noor and Nandakumar, Karthik and Lukas, Nils},
  booktitle = {1st Workshop on GenAI Watermarking, ICLR 2025},
  year      = {2025}
}
```
