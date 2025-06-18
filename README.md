<div align="center">

<img src="assets/banner.webp" alt="LTX-Video Community Trainer Banner" width="75%">

[Official GitHub Repo](https://github.com/Lightricks/LTX-Video) |
[Website](https://www.lightricks.com/ltxv) |
[Model](https://huggingface.co/Lightricks/LTX-Video) |
[Demo](https://app.ltx.studio/ltx-video) |
[Paper](https://arxiv.org/abs/2501.00103) |
[Discord](https://discord.gg/Mn8BRgUKKy)

</div>

This repository provides tools and scripts for training and fine-tuning Lightricks' [LTX-Video (LTXV)](https://github.com/Lightricks/LTX-Video) model.
It allows training LoRAs on top of LTX-Video, as well as fine-tuning the entire model on custom datasets.
The repository also includes auxiliary utilities for preprocessing datasets, captioning videos, splitting scenes, etc.

---

<div align="center">

| <img src="assets/cakeify.gif" width="256px">  | <img src="assets/squish.gif" width="256px"> |
| --------------------------------------------- | ------------------------------------------- |
| <img src="assets/dissolve.gif" width="256px"> | <img src="assets/slime.gif" width="256px">  |

<small>Examples of effects trained as LoRAs on top of LTX-Video 13B</small>

</div>

---

## 📚 Table of Contents

- [Getting Started](#-getting-started)
- [Dataset Preparation](#-dataset-preparation)
  - [Split Scenes](#1-split-scenes-split_scenespy)
  - [Caption Videos](#2-caption-videos-caption_videospy)
  - [Preprocess Dataset](#3-dataset-preprocessing-preprocess_datasetpy)
- [Training Configuration](#-training-configuration)
  - [Example Configurations](#example-configurations)
- [Training Modes](#-training-modes)
  - [Standard LoRA Training](#standard-lora-training)
  - [Full Model Fine-tuning](#full-model-fine-tuning)
  - [In-Context LoRA (IC-LoRA) Training](#in-context-lora-ic-lora-training)
- [Running the Trainer](#-running-the-trainer)
- [Quick Start with Gradio](#-quick-start-with-gradio)
- [Using Trained LoRAs in ComfyUI](#-using-trained-loras-in-comfyui)
- [Example LoRAs](#-example-loras)
- [Contributing](#-contributing)
- [Acknowledgements](#-acknowledgements)

---

## 📝 Changelog

- **06.05.2025:** Added support for LTXV 13B. Example training configs can be found in
  [configs/ltxv_13b_lora_cakeify.yaml](configs/ltxv_13b_lora_cakeify.yaml) and [configs/ltxv_13b_lora_squish.yaml](configs/ltxv_13b_lora_squish.yaml).

> [!NOTE]
> When training with the LTXV 13B model:
>
> The training process is the same as with the 2B model, but you must enable gradient checkpointing in your configuration (`enable_gradient_checkpointing: true`).
> We recommend using the reference configuration files provided in the `configs` directory as starting points.

---

## 🚀 Getting Started

First, install [uv](https://docs.astral.sh/uv/getting-started/installation/) if you haven't already.
Then clone the repository and install the dependencies:

```bash
git clone https://github.com/Lightricks/LTX-Video-Trainer
cd LTX-Video-Trainer
uv sync
source .venv/bin/activate
```

Follow the steps below to prepare your dataset and configure your training job.

---

### 🎬 Dataset Preparation

This section describes the workflow for preparing and preprocessing your dataset for training. The general flow is:

1. (Optional) Split long videos into scenes using `split_scenes.py`
2. (Optional) Generate captions for your videos using `caption_videos.py`
3. Preprocess your dataset using `preprocess_dataset.py` to compute and cache video latents and text embeddings
4. Run the trainer with your preprocessed dataset

#### 1. Split Scenes (`split_scenes.py`)

If you're starting with long-form videos (e.g., downloaded from YouTube), you should first split them into shorter, coherent scenes:

```bash
# Split a long video into scenes
python scripts/split_scenes.py input.mp4 scenes_output_dir/ \
    --filter-shorter-than 5s
```

This will create multiple video clips in `scenes_output_dir`.
These clips will be the input for the captioning step, if you choose to use it.

#### 2. Caption Videos (`caption_videos.py`)

If your dataset doesn't include captions, you can automatically generate them using vision-language models.
Use the directory containing your video clips (either from step 1, or your own clips):

```bash
# Generate captions for all videos in the scenes directory
python scripts/caption_videos.py scenes_output_dir/ \
    --output scenes_output_dir/captions.json
```

If you're running into VRAM issues, try enabling 8-bit quantization to reduce memory usage:
```bash
python scripts/caption_videos.py scenes_output_dir/ \
    --output scenes_output_dir/captions.json \
    --use-8bit
```

This will create a captions.json file which contains video paths and their captions.
This JSON file will be used as input for the data preprocessing step.

By default, the script uses the Qwen2.5-VL model for media captioning.

#### 3. Dataset Preprocessing (`preprocess_dataset.py`)

This step preprocesses your video dataset by:

1. Resizing and cropping videos to fit specified resolution buckets
2. Computing and caching video latent representations
3. Computing and caching text embeddings for captions

Using the captions.json file generated in step 2:

```bash
# Preprocess the dataset using the generated captions.json
python scripts/preprocess_dataset.py scenes_output_dir/captions.json \
    --resolution-buckets "768x768x25" \
    --caption-column "caption" \
    --video-column "media_path"
```

The preprocessing significantly accelerates training and reduces GPU memory usage.

##### Resolution Buckets

Videos are organized into "buckets" of specific dimensions (width × height × frames). Each video is assigned to the nearest matching bucket.
Currently, the trainer only supports using a single resolution bucket.

The dimensions of each bucket must follow these constraints due to LTX-Video's VAE architecture:

- Spatial dimensions (width and height) must be multiples of 32
- Number of frames must be a multiple of 8 plus 1 (e.g., 25, 33, 41, 49, etc.)

Guidelines for choosing training resolution:

- For high-quality, detailed videos: use larger spatial dimensions (e.g. 768x448) with fewer frames (e.g. 89)
- For longer, motion-focused videos: use smaller spatial dimensions (512×512) with more frames (121)
- Memory usage increases with both spatial and temporal dimensions

Example usage:

```bash
python scripts/preprocess_dataset.py /path/to/dataset \
    --resolution-buckets "768x768x25"
```

This creates a bucket with:

- 768×768 resolution
- 25 frames

Videos are processed as follows:

1. Videos are resized maintaining aspect ratio until either width or height matches the target (768 in this example)
2. The larger dimension is center cropped to match the bucket's dimensions
3. Frames are sampled uniformly to match the bucket's frame count (25 in this example)

> [!NOTE]
> The sequence length processed by the transformer model can be calculated as:
>
> ```
> sequence_length = (H/32) * (W/32) * ((F-1)/8 + 1)
> ```
>
> Where:
>
> - H = Height of video's latent
> - W = Width of video's latent
> - F = Number of latent frames
> - 32 = VAE's spatial downsampling factor
> - 8 = VAE's temporal downsampling factor
>
> For example, a 768×448×89 video would have sequence length:
>
> ```
> (768/32) * (448/32) * ((89-1)/8 + 1) = 24 * 14 * 12 = 4,032
> ```
>
> Keep this in mind when choosing video dimensions, as longer sequences require more memory and computation power.

> [!WARNING]
> While the preprocessing script supports multiple buckets, the trainer currently only works with a single resolution bucket.
> Please ensure you specify just one bucket in your preprocessing command.

##### Dataset Format

The trainer supports either videos or single images.
Note that your dataset must be homogeneous - either all videos or all images, mixing is not supported.
When using images, follow the same preprocessing steps and format requirements as with videos,
simply provide image files instead of video files.

The dataset must be a CSV, JSON, or JSONL metadata file with columns for captions and video paths:

```bash
# Using CSV/JSON/JSONL metadata file
python scripts/preprocess_dataset.py dataset.json \
    --caption-column "caption" \
    --video-column "video_path" \
    --model-source "LTXV_13B_097_DEV"  # Optional: specify a specific version, defaults to latest
```

##### Output Structure

The preprocessed data is saved in a `.precomputed` directory:

```
dataset/
└── .precomputed/
    ├── latents/           # Cached video latents
    ├── conditions/        # Cached text embeddings
    └── reference_latents/ # Cached reference video latents (for IC-LoRA training)
```

##### IC-LoRA Reference Video Preprocessing

For IC-LoRA training, you can preprocess datasets that include reference videos. Reference videos provide clean conditioning input while target videos represent the desired transformed output.

**Dataset Format with Reference Videos:**

JSON/JSONL format:
```json
[
  {
    "caption": "A video of a cake being decorated with CAKEIFY effect",
    "media_path": "videos/target_001.mp4",
    "reference_path": "videos/reference_001.mp4"
  }
]
```

CSV format:
```csv
caption,media_path,reference_path
"A video of a cake being decorated with CAKEIFY effect","videos/target_001.mp4","videos/reference_001.mp4"
```

**Preprocessing with Reference Videos:**

```bash
# Using JSON/JSONL dataset
python scripts/preprocess_dataset.py dataset.json \
    --resolution-buckets "768x768x25" \
    --caption-column "caption" \
    --video-column "media_path" \
    --reference-column "reference_path"

# Using CSV dataset
python scripts/preprocess_dataset.py dataset.json \
    --resolution-buckets "768x768x25" \
    --caption-column "caption" \
    --video-column "media_path" \
    --reference-column "reference_path"
```

This will create an additional `reference_latents/` directory containing the preprocessed reference video latents.

##### LoRA Trigger Words

When training a LoRA, you can specify a trigger token that will be prepended to all captions:

```bash
python scripts/preprocess_dataset.py /path/to/dataset \
    --resolution-buckets "1024x576x65" \
    --id-token "<TOK>"
```

This acts as a trigger word that activates the LoRA during inference when you include the same token in your prompts.

##### Decoding videos

By providing the `--decode-videos` flag, the script will also VAE-decode the precomputed latents and save the resulting videos under `.precomputed/decoded_videos` so you can look at and evaluate the data in the latents. This is useful for debugging and ensuring that your dataset is being processed correctly.

```bash
# Preprocess dataset and decode videos for verification
python scripts/preprocess_dataset.py /path/to/dataset \
    --resolution-buckets "768x768x25" \
    --decode-videos

# For IC-LoRA datasets, this will also decode reference videos
python scripts/preprocess_dataset.py dataset.json \
    --resolution-buckets "768x768x25" \
    --reference-column "reference_path" \
    --decode-videos
```

For single-frame images, they are saved as PNG files instead of MP4.

## ⚙️ Training Configuration

The trainer uses structured Pydantic models for configuration, making it easy to customize training parameters.
The main configuration class is [`LtxvTrainerConfig`](src/ltxv_trainer/config.py), which includes:

- **ModelConfig**: Base model and training mode settings
- **LoraConfig**: LoRA fine-tuning parameters
- **OptimizationConfig**: Learning rates, batch sizes, and scheduler settings
- **AccelerationConfig**: Mixed precision and optimization settings
- **DataConfig**: Data loading parameters
- **ValidationConfig**: Validation and inference settings
- **CheckpointsConfig**: Checkpoint saving frequency and retention settings
- **FlowMatchingConfig**: Timestep sampling parameters
- **ConditioningConfig**: Video conditioning settings (reference videos, first frame conditioning)

### Example Configurations

Check out our example configurations in the `configs` directory. You can use these as templates
for your training runs:

- 📄 [Full Model Fine-tuning Example](configs/ltxv_2b_full.yaml)
- 📄 [LoRA Fine-tuning Example](configs/ltxv_2b_lora.yaml)
- 📄 [LoRA Fine-tuning Example (Low VRAM)](configs/ltxv_2b_lora_low_vram.yaml) - Optimized for GPUs with 24GB VRAM.
- 📄 [IC-LoRA Training Examples](configs/ltxv_13b_ic_lora.yaml) - Video-to-video transformation training

---

## 🎯 Training Modes

The trainer supports three main training modes, each suited for different use cases:

### Standard LoRA Training

Standard LoRA (Low-Rank Adaptation) training fine-tunes the model by adding small, trainable adapter layers while keeping the base model frozen. This approach:

- Requires significantly less memory and compute than full fine-tuning
- Produces small, portable weight files
- Is ideal for learning specific styles, effects, or concepts
- Can be easily combined with other LoRAs during inference

Configure standard LoRA training with:
```yaml
model:
  training_mode: "lora"
conditioning:
  mode: "none"
```

### Full Model Fine-tuning

Full model fine-tuning updates all parameters of the base model, providing maximum flexibility but requiring substantial computational resources:

- Offers the highest potential quality and capability improvements
- Requires significant GPU memory (typically 40GB+ for larger models)
- Produces large checkpoint files (several GB)
- Best for major model adaptations or when LoRA limitations are reached

Configure full fine-tuning with:
```yaml
model:
  training_mode: "full"
conditioning:
  mode: "none"
```

### In-Context LoRA (IC-LoRA) Training

IC-LoRA is a specialized training mode for video-to-video transformations. Unlike standard training modes that learn from individual videos, IC-LoRA learns transformations by comparing pairs of videos:

- **Reference videos** provide clean, unnoised conditioning input showing the "before" state
- **Target videos** are noised during training and represent the desired "after" state
- The model learns transformations from reference videos to target videos
- Loss is applied only to the target portion, not the reference
- Enables learning complex video-to-video effects like style transfers, control adapters, etc.

To enable IC-LoRA training, configure your YAML file with:

```yaml
model:
  training_mode: "lora"
conditioning:
  mode: "reference_video"
  reference_latents_dir: "reference_latents"  # Directory name for reference video latents
```

**Dataset Requirements for IC-LoRA:**
- Your dataset must contain paired videos where each target video has a corresponding reference video
- Reference and target videos must have *identical* resolution and length
- Both reference and target videos should be preprocessed together using the same resolution buckets

**Configuration Requirements for IC-LoRA:**
- You must provide `reference_videos` in your validation configuration when using IC-LoRA training
- The number of reference videos must match the number of validation prompts

Example validation configuration for IC-LoRA:
```yaml
validation:
  prompts:
    - "First prompt"
    - "Second prompt"
  reference_videos:
    - "path/to/reference1.mp4"
    - "path/to/reference2.mp4"
```


---

## ⚡ Running the Trainer

After preprocessing your dataset and preparing a configuration file, you can start training using the trainer script:

```bash
# Train a LoRA
python scripts/train.py configs/ltxv_2b_lora.yaml

# Fine-tune the full model
python scripts/train.py configs/ltxv_2b_full.yaml
```

The trainer loads your configuration, initializes models, applies optimizations,
runs the training loop with progress tracking, generates validation videos (if configured), and saves the trained weights.

For LoRA training, the weights will be saved as `lora_weights.safetensors` in your output directory.
For full model fine-tuning, the weights will be saved as `model_weights.safetensors`.

### 🖥️ Distributed / Multi-GPU Training

For larger training jobs, you can run the trainer across multiple GPUs on a single machine using our
distributed training script, which leverages [Hugging Face Accelerate](https://huggingface.co/docs/accelerate/index).

Use the provided script:

```bash
python scripts/train_distributed.py CONFIG_PATH [OPTIONS]
```

**Example:**

```bash
# Launch distributed training on all available GPUs
python scripts/train_distributed.py configs/ltxv_2b_lora.yaml

# Specify the number of processes/GPUs explicitly
CUDA_VISIBLE_DEVICES=0,1 python scripts/train_distributed.py configs/ltxv_2b_lora.yaml --num_processes 2
```

**Options:**
- `--num_processes`: Number of GPUs/processes to use (overrides auto-detection).
- `--disable_progress_bars`: Disables rich progress bars (recommended for multi-GPU runs).

The script will automatically detect the number of available GPUs using `nvidia-smi` if `--num_processes` is not specified.

### 🤗 Pushing Models to Hugging Face Hub

You can automatically push your trained models to the Hugging Face Hub by adding the following to your configuration YAML:

```yaml
hub:
  push_to_hub: true
  hub_model_id: "your-username/your-model-name"  # Your HF username and desired repo name
```

Before pushing, make sure you:
1. Have a Hugging Face account
2. Are logged in via `huggingface-cli login` or have set the `HUGGING_FACE_HUB_TOKEN` environment variable
3. Have write access to the specified repository (it will be created if it doesn't exist)

The trainer will:
- Create a model card with training details and sample outputs
- Upload the model weights (both original and ComfyUI-compatible versions)
- Push sample videos as GIFs in the model card
- Include training configuration and prompts

---

## Fast and simple: Running the Complete Pipeline as one command

For a streamlined experience, you can use `run_pipeline.py` which automates the entire training workflow. For example provide it with a template configuration which will be instantiated based on the provided values and media files.

```bash
python scripts/run_pipeline.py [LORA_BASE_NAME] \
    --resolution-buckets "768x768x49" \
    --config-template configs/ltxv_2b_lora_template.yaml \
    --rank 32
```

This script will:

1. Process raw videos in `[basename]_raw/` directory (if they exist):

   - Split long videos into scenes
   - Save scenes to `[basename]_scenes/`

2. Generate captions for the scenes (if scenes exist):

   - Uses Qwen-2.5-VL for captioning
   - Saves captions to `[basename]_scenes/captions.json`

3. Preprocess the dataset:

   - Computes and caches video latents
   - Computes and caches text embeddings
   - Decodes videos for verification

4. Run the training:

   - Uses the provided config template
   - Automatically extracts validation prompts from captions
   - Saves the final model weights

5. Convert LoRA to ComfyUI format:
   - Automatically converts the trained LoRA weights to ComfyUI format
   - Saves the converted weights with "_comfy" suffix

Required arguments:

- `basename`: Base name for your project (e.g., "slime")
- `--resolution-buckets`: Video resolution in format "WxHxF" (e.g., "768x768x49")
- `--config-template`: Path to your configuration template file
- `--rank`: LoRA rank (1-128) for training

The script will create the following directory structure:

```
[basename]_raw/          # Place your raw videos here
[basename]_scenes/       # Split scenes and captions
└── .precomputed/       # Preprocessed data
    ├── latents/       # Cached video latents
    ├── conditions/    # Cached text embeddings
    └── decoded_videos/ # Decoded videos for verification
outputs/                # Training outputs and checkpoints
    └── lora_weights_comfy.safetensors  # ComfyUI-compatible LoRA weights
```

## 🎨 Quick Start with Gradio

If you prefer a graphical interface, you can also use our Gradio web UI:

```bash
# Install dependencies if you haven't already
uv sync
source .venv/bin/activate

# Launch the Gradio interface
python app_gradio.py
```

This will open a web interface at `http://localhost:7860` that provides all training functionality in a user-friendly way.

- Upload training videos
- Generate or edit captions
- Configure training parameters
- Monitor training progress
- View validation samples in real-time
- Download trained models

The interface provides the same functionality as the command-line tools but in a more intuitive way.

---

## 🔌 Using Trained LoRAs in ComfyUI

After training your LoRA, you can use it in ComfyUI by following these steps:

1. Convert your trained LoRA weights to ComfyUI format using the conversion script:

   ```bash
   python scripts/convert_checkpoint.py your_lora_weights.safetensors --to-comfy
   ```

2. Copy the converted LoRA weights (`.safetensors` file) to the `models/loras` folder in your ComfyUI installation.

3. In your ComfyUI workflow:
   - Use the built-in "Load LoRA" node to load your LoRA file
   - Connect it to your LTXV nodes to apply the LoRA to your generation

You can find reference Text-to-Video (T2V) and Image-to-Video (I2V) workflows in the [official LTXV ComfyUI repository](https://github.com/Lightricks/ComfyUI-LTXVideo).

---

## 🍰 Example LoRAs

Here are some example LoRAs trained using this trainer, along with their training datasets:

### Cakeify Effect

The [Cakeify LoRA](https://huggingface.co/Lightricks/LTX-Video-Cakeify-LoRA) transforms videos to make objects appear as if they're made of cake.
The effect was trained on the [Cakeify Dataset](https://huggingface.co/datasets/Lightricks/Cakeify-Dataset).

### Squish Effect

The [Squish LoRA](https://huggingface.co/Lightricks/LTX-Video-Squish-LoRA) creates a playful squishing effect on objects in videos.
It was trained on the [Squish Dataset](https://huggingface.co/datasets/Lightricks/Squish-Dataset), which contains just 5 example videos.

These examples demonstrate how you can train specialized video effects using this trainer.
Feel free to use these datasets as references for preparing your own training data.

---

## ️🔧 Utility Scripts

### LoRA Format Converter

Using `scripts/convert_checkpoint.py` you can convert your LoRA saved file from `diffusers` library format to `ComfyUI` format.

```bash
# Convert from diffusers to ComfyUI format
python scripts/convert_checkpoint.py input.safetensors --to-comfy --output_path output.safetensors

# Convert from ComfyUI to diffusers format
python scripts/convert_checkpoint.py input.safetensors --output_path output.safetensors
```

If no output path is specified, the script will automatically generate one by adding `_comfy` or `_diffusers` suffix to the input filename.

### Latents Decoding Script

Using `scripts/decode_latents.py` you can decode precomputed video latents back into video files.
This is useful for verifying the quality of your preprocessed dataset or debugging the preprocessing pipeline.

```bash
# Basic usage
python scripts/decode_latents.py /path/to/latents/dir --output-dir /path/to/output
```

The script will:

1. Load the VAE model from the specified path
2. Process all `.pt` latent files in the input directory
3. Decode each latent back into a video using the VAE
4. Save the resulting videos as MP4 files in the output directory

---

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

- **Share Your Work**: If you've trained interesting LoRAs or achieved cool results, please share them with the community.
- **Report Issues**: Found a bug or have a suggestion? Open an issue on GitHub.
- **Submit PRs**: Help improve the codebase with bug fixes or general improvements.
- **Feature Requests**: Have ideas for new features? Let us know through GitHub issues.

---

## 🫶 Acknowledgements

Parts of this project are inspired by and incorporate ideas from several awesome open-source projects:

- [a-r-r-o-w/finetrainers](https://github.com/a-r-r-o-w/finetrainers)
- [bghira/SimpleTuner](https://github.com/bghira/SimpleTuner)

---

Happy training! 🎉
