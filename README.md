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

- **23.06.2025:** Added support for training IC-LoRAs (In-Context LoRAs) for advanced video-to-video transformations.
- **06.05.2025:** Added support for LTXV 13B. Example training configs can be found in [configs/ltxv_13b_lora_cakeify.yaml](configs/ltxv_13b_lora_cakeify.yaml) and [configs/ltxv_13b_lora_squish.yaml](configs/ltxv_13b_lora_squish.yaml).

---

## 🍰 Example LoRAs

- [Cakeify LoRA](https://huggingface.co/Lightricks/LTX-Video-Cakeify-LoRA): Transforms videos to make objects appear as if they're made of cake. ([Cakeify Dataset](https://huggingface.co/datasets/Lightricks/Cakeify-Dataset))
- [Squish LoRA](https://huggingface.co/Lightricks/LTX-Video-Squish-LoRA): Creates a playful squishing effect. ([Squish Dataset](https://huggingface.co/datasets/Lightricks/Squish-Dataset))

These examples demonstrate how you can train specialized video effects using this trainer. Use these datasets as references for preparing your own training data.

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
