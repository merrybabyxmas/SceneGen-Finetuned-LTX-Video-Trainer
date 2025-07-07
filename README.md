<div align="center">

<img src="assets/banner.webp" alt="LTX-Video Community Trainer Banner" width="75%">

[Official GitHub Repo](https://github.com/Lightricks/LTX-Video) |
[Website](https://www.lightricks.com/ltxv) |
[Model](https://huggingface.co/Lightricks/LTX-Video) |
[Demo](https://app.ltx.studio/ltx-video) |
[Paper](https://arxiv.org/abs/2501.00103) |
[Discord](https://discord.gg/Mn8BRgUKKy)

</div>


This repository provides tools and scripts for training and fine-tuning Lightricks' [LTX-Video (LTXV)](https://github.com/Lightricks/LTX-Video) model. It enables LoRA training, full fine-tuning, and video-to-video transformation workflows on custom datasets.

---

<div align="center">

| <img src="assets/cakeify.gif" width="256px">  | <img src="assets/squish.gif" width="256px"> |
| --------------------------------------------- | ------------------------------------------- |
| <img src="assets/dissolve.gif" width="256px"> | <img src="assets/slime.gif" width="256px">  |

<small>Examples of effects trained as LoRAs on top of LTX-Video 13B</small>

</div>

---

## 📖 Documentation

All detailed guides and technical documentation have been moved to the `docs/` directory:

- [⚡ Quick Start Guide](docs/quick-start.md)
- [🎬 Dataset Preparation](docs/dataset-preparation.md)
- [🛠️ Training Modes](docs/training-modes.md)
- [⚙️ Configuration Reference](docs/configuration-reference.md)
- [🚀 Training Guide](docs/training-guide.md)
- [🔧 Utility Scripts](docs/utility-scripts.md)
- [🛡️ Troubleshooting Guide](docs/troubleshooting.md)

---

## 🔥 Changelog

- **08.07.2025:** Added support for training IC-LoRAs (In-Context LoRAs) for advanced video-to-video transformations. See the [training modes](https://github.com/LightricksResearch/LTX-Video-Trainer-Internal/blob/main/docs/training-modes.md#-in-context-lora-ic-lora-training) doc for more details.
Pretrained control models: [Depth](https://huggingface.co/Lightricks/LTX-Video-ICLoRA-depth-13b-0.9.7), [Pose](https://huggingface.co/Lightricks/LTX-Video-ICLoRA-pose-13b-0.9.7), [Canny](https://huggingface.co/Lightricks/LTX-Video-ICLoRA-canny-13b-0.9.7).
- **06.05.2025:** Added support for LTXV 13B.
  An example training configuration can be found [here](configs/ltxv_13b_lora_cakeify.yaml).

---

## 🍰 Example Models

### Standard LoRAs
- [Cakeify LoRA](https://huggingface.co/Lightricks/LTX-Video-Cakeify-LoRA): Transforms videos to make objects appear as if they're made of cake. ([Dataset](https://huggingface.co/datasets/Lightricks/Cakeify-Dataset))
- [Squish LoRA](https://huggingface.co/Lightricks/LTX-Video-Squish-LoRA): Creates a playful squishing effect. ([Dataset](https://huggingface.co/datasets/Lightricks/Squish-Dataset))

### IC-LoRA Control Adapters
- [Depth Map Control](https://huggingface.co/Lightricks/LTX-Video-ICLoRA-depth-13b-0.9.7): Generate videos from depth maps.
- [Human Pose Control](https://huggingface.co/Lightricks/LTX-Video-ICLoRA-pose-13b-0.9.7): Generate videos from pose skeletons.
- [Canny Edge Control](https://huggingface.co/Lightricks/LTX-Video-ICLoRA-canny-13b-0.9.7): Generate videos from Canny edge maps. ([Canny Control Dataset](https://huggingface.co/datasets/Lightricks/Canny-Control-Dataset))

These examples demonstrate how you can train specialized video effects and control adapters using this trainer. Use these datasets as references for preparing your own training data.

---

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

- **Share Your Work**: If you've trained interesting LoRAs or achieved cool results, please share them with the community.
- **Report Issues**: Found a bug or have a suggestion? Open an issue on GitHub.
- **Submit PRs**: Help improve the codebase with bug fixes or general improvements.
- **Feature Requests**: Have ideas for new features? Let us know through GitHub issues.

---

## 💬 Join the Community

Have questions, want to share your results, or need real-time help?

Join our [community Discord server](https://discord.gg/Mn8BRgUKKy) to connect with other users and the development team!

- Get troubleshooting help
- Share your training results and workflows
- Collaborate on new ideas and features
- Stay up to date with announcements and updates

We look forward to seeing you there!

---

## 🫶 Acknowledgements

Parts of this project are inspired by and incorporate ideas from several awesome open-source projects:

- [a-r-r-o-w/finetrainers](https://github.com/a-r-r-o-w/finetrainers)
- [bghira/SimpleTuner](https://github.com/bghira/SimpleTuner)

---

Happy training! 🎉
