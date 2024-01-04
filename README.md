# CLIP Papers

This repository contains a comprehensive collection of the most important papers related to contrastive pretraining for vision, language, and audio. The papers are organized categorically, and sorted by year and month of publication.

## Contrastive Language-Image Pretraining (CLIP)

The following table contains a list of papers that are directly related to CLIP, or that extend CLIP in some way, such as by improving the training process, or by changing the data filtering process. Every entry in this table is distinguished by contrastive learning being the *primary* pretraining objective, as opposed to models than employ multiple pretraining objectives, combining contrastive learning with other pretraining objectives masked language modeling (MLM).

<!--- PURE_CLIP_TABLE -->
| **Model** | **Year** | **Month** | **Paper Title** | **Novel Development** | **Arxiv** | **Github** | **Open Source** | **License** | **Model Card** | **OpenCLIP Integration** |
|:---------:|:---------:|:--------:|:----------------:|:----------------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
| CLIP | 2021 | 2 | Learning Transferable Visual Models From Natural Language Supervision | Simplified Contrastive Language-Image Pretraining | [![arXiv](https://img.shields.io/badge/arXiv-2103.00020-b31b1b.svg)](https://arxiv.org/abs/2103.00020) | [![GitHub](https://img.shields.io/github/stars/openai/CLIP?style=social)](https://github.com/openai/CLIP/) | ✔️ | [License](https://github.com/openai/CLIP/blob/main/LICENSE) | [Model Card](https://github.com/openai/CLIP/blob/main/model-card.md) | ✔️ |
| ALIGN | 2021 | 2 | Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision | Extend from captions to noisy alt-text to avoid expensive filtering and post-processing | [![arXiv](https://img.shields.io/badge/arXiv-2102.05918-b31b1b.svg)](https://arxiv.org/abs/2102.05918) |  | ✔️ |  | [Model Card](https://huggingface.co/kakaobrain/align-base) | ❌ |
| CLOOB | 2021 | 10 | CLOOB: Modern Hopfield Networks with InfoLOOB Outperform CLIP | Avoid saturation of InfoNCE objective | [![arXiv](https://img.shields.io/badge/arXiv-2110.11316-b31b1b.svg)](https://arxiv.org/abs/2110.11316) | [![GitHub](https://img.shields.io/github/stars/ml-jku/cloob?style=social)](https://github.com/ml-jku/cloob) | ✔️ | [License](https://github.com/ml-jku/cloob?tab=readme-ov-file#license) |  | ❌ |
| DeCLIP | 2021 | 10 | Supervision Exists Everywhere: A Data Efficient Contrastive Language-Image Pre-training Paradigm | Data efficiency through supervision | [![arXiv](https://img.shields.io/badge/arXiv-2110.05208-b31b1b.svg)](https://arxiv.org/abs/2110.05208) | [![GitHub](https://img.shields.io/github/stars/Sense-GVT/DeCLIP?style=social)](https://github.com/Sense-GVT/DeCLIP) | ✔️ | [License](https://github.com/Sense-GVT/DeCLIP?tab=readme-ov-file#license) |  | ❌ |
| FILIP | 2021 | 11 | FILIP: Fine-grained Interactive Language-Image Pre-Training | Adds token-wise maximum similarity bewteen visual and textual features for efficient and fine-grained semantic alignment  | [![arXiv](https://img.shields.io/badge/arXiv-2111.07783-b31b1b.svg)](https://arxiv.org/abs/2111.07783) |  | ✔️ |  |  | ❌ |
| DeFILIP | 2022 | 3 | Democratizing Contrastive Language-Image Pre-training: A CLIP Benchmark of Data, Model, and Supervision | Combines DeCLIP and FILIP | [![arXiv](https://img.shields.io/badge/arXiv-2203.05796-b31b1b.svg)](https://arxiv.org/abs/2203.05796) | [![GitHub](https://img.shields.io/github/stars/Sense-GVT/DeCLIP?style=social)](https://github.com/Sense-GVT/DeCLIP) | ✔️ | [License](https://github.com/Sense-GVT/DeCLIP?tab=readme-ov-file#license) |  | ❌ |
| PyramidCLIP | 2022 | 4 | PyramidCLIP: Hierarchical Feature Alignment for Vision-language Model Pretraining | Relax assumption that image and metadata are in one-to-one correspondence | [![arXiv](https://img.shields.io/badge/arXiv-2204.14095-b31b1b.svg)](https://arxiv.org/abs/2204.14095) |  | ❌ |  |  | ❌ |
| KLITE | 2022 | 4 | K-LITE: Learning Transferable Visual Models with External Knowledge | Augment caption text with external knowledge | [![arXiv](https://img.shields.io/badge/arXiv-2204.09222-b31b1b.svg)](https://arxiv.org/abs/2204.09222) | [![GitHub](https://img.shields.io/github/stars/microsoft/klite?style=social)](https://github.com/microsoft/klite) | ✔️ | [License](https://github.com/microsoft/klite/blob/main/LICENSE) |  | ❌ |
| CyCLIP | 2022 | 5 | CyCLIP: Cyclic Contrastive Language-Image Pretraining | Formalize and optimize for geometric consistency in image and text spaces | [![arXiv](https://img.shields.io/badge/arXiv-2205.14459-b31b1b.svg)](https://arxiv.org/abs/2205.14459) | [![GitHub](https://img.shields.io/github/stars/goel-shashank/CyCLIP?style=social)](https://github.com/goel-shashank/CyCLIP) | ✔️ | [License](https://github.com/goel-shashank/CyCLIP?tab=readme-ov-file#licenses) |  | ❌ |
| FLIP | 2022 | 12 | Scaling Language-Image Pre-training via Masking | Masking images prior to encoding improves speed-accuracy trade-off for CLIP | [![arXiv](https://img.shields.io/badge/arXiv-2212.00794-b31b1b.svg)](https://arxiv.org/abs/2212.00794) | [![GitHub](https://img.shields.io/github/stars/facebookresearch/flip?style=social)](https://github.com/facebookresearch/flip) | ✔️ | [License](https://github.com/facebookresearch/flip/blob/main/LICENSE) |  | ❌ |
| OpenCLIP | 2022 | 12 | Reproducible scaling laws for contrastive language-image learning | Open-source implementation of CLIP | [![arXiv](https://img.shields.io/badge/arXiv-2212.07143-b31b1b.svg)](https://arxiv.org/abs/2212.07143) | [![GitHub](https://img.shields.io/github/stars/mlfoundations/open_clip?style=social)](https://github.com/mlfoundations/open_clip) | ✔️ | [License](https://github.com/mlfoundations/open_clip/blob/main/LICENSE) | [Model Card](https://github.com/mlfoundations/open_clip/blob/main/docs/PRETRAINED.md) | ✔️ |
| EVA-CLIP | 2023 | 3 | EVA-CLIP: Improved Training Techniques for CLIP at Scale | Improved representation learning, optimization, and augmentation for faster training | [![arXiv](https://img.shields.io/badge/arXiv-2303.15389v1-b31b1b.svg)](https://arxiv.org/abs/2303.15389v1) | [![GitHub](https://img.shields.io/github/stars/baaivision/EVA?style=social)](https://github.com/baaivision/EVA/tree/master/EVA-CLIP) | ✔️ |  | [Model Card](https://github.com/baaivision/EVA/tree/master/EVA-CLIP#model-card) | ✔️ |
| SigLIP | 2023 | 3 | Sigmoid Loss for Language Image Pre-Training | Sigmoid loss allows disentangling loss from batch size | [![arXiv](https://img.shields.io/badge/arXiv-2303.15343-b31b1b.svg)](https://arxiv.org/abs/2303.15343) | [![GitHub](https://img.shields.io/github/stars/google-research/big_vision?style=social)](https://github.com/google-research/big_vision) | ✔️ | [License](https://github.com/google-research/big_vision?tab=readme-ov-file#license) |  | ✔️ |
| CLIPA | 2023 | 5 | An Inverse Scaling Law for CLIP Training | Insight into relationship between encoder size and training input sequence lengths leads to more efficient training   | [![arXiv](https://img.shields.io/badge/arXiv-2305.07017-b31b1b.svg)](https://arxiv.org/abs/2305.07017) | [![GitHub](https://img.shields.io/github/stars/UCSC-VLAA/CLIPA?style=social)](https://github.com/UCSC-VLAA/CLIPA) | ✔️ | [License](https://github.com/UCSC-VLAA/CLIPA#license) |  | ✔️ |
| MetaCLIP | 2023 | 9 | Demystifying CLIP Data | Rigorous study to reveal CLIP's data curation process | [![arXiv](https://img.shields.io/badge/arXiv-2309.16671-b31b1b.svg)](https://arxiv.org/abs/2309.16671) | [![GitHub](https://img.shields.io/github/stars/facebookresearch/MetaCLIP?style=social)](https://github.com/facebookresearch/MetaCLIP) | ✔️ | [License](https://github.com/facebookresearch/MetaCLIP/blob/main/LICENSE) |  | ✔️ |
| DFN | 2023 | 11 | Data Filtering Networks | A model trained on high-quality data can be used to filter massive online data employed to train the final CLIP model | [![arXiv](https://img.shields.io/badge/arXiv-2309.17425-b31b1b.svg)](https://arxiv.org/abs/2309.17425) |  | ✔️ | [License](https://huggingface.co/apple/DFN5B-CLIP-ViT-H-14-378/blob/main/LICENSE) | [Model Card](https://huggingface.co/apple/DFN5B-CLIP-ViT-H-14-378) | ✔️ |
<!--- PURE_CLIP_TABLE -->

## CLIP + Additional Pretraining Objectives

Models that extend CLIP by adding additional pretraining objectives, such as masked language modeling (MLM).

The acronyms used in the table below are as follows:

- **DR**: Dataset Reinforcement
- **H-ITC**: Hierarchical Image-Text Contrastive
- **ISS**: Image Self-Supervision
- **ITM**: Image-Text Matching
- **LM**: Language Modeling
- **MIM**: Masked Image Modeling
- **MLM**: Masked Language Modeling
- **MMM**: Masked Multimodal Modeling
- **MSD**: Masked Self-Distillation

All models in this table also use CLIP-style contrastive learning as a pretraining objective.


<!--- CLIP_PLUS_TABLE -->
| **Model** | **Year** | **Month** | **Paper Title** | **Pretraining Techniques** | **Arxiv** | **Github** | **Open Source** | **License** |
|:---------:|:---------:|:--------:|:----------------:|:----------------:|:--------:|:--------:|:--------:|:--------:|
| SLIP | 2021 | 12 | SLIP: Self-supervision meets Language-Image Pre-training | ISS | [![arXiv](https://img.shields.io/badge/arXiv-2112.1275-b31b1b.svg)](https://arxiv.org/abs/2112.1275) | [![GitHub](https://img.shields.io/github/stars/facebookresearch/SLIP?style=social)](https://github.com/facebookresearch/SLIP) | ✔️ | [License](https://github.com/facebookresearch/SLIP/blob/main/LICENSE) |
| FLAVA | 2021 | 12 | FLAVA: A Foundational Language And Vision Alignment Model | ITM+MMM+MIM+MLM | [![arXiv](https://img.shields.io/badge/arXiv-2112.04482-b31b1b.svg)](https://arxiv.org/abs/2112.04482) | [![GitHub](https://img.shields.io/github/stars/facebookresearch/multimodal?style=social)](https://github.com/facebookresearch/multimodal/tree/main/examples/flava) | ✔️ | [License](https://huggingface.co/facebook/flava-full#:~:text=License%3A,bsd%2D3%2Dclause) |
| BLIP | 2022 | 1 | BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation | ITM+LM | [![arXiv](https://img.shields.io/badge/arXiv-2201.12086-b31b1b.svg)](https://arxiv.org/abs/2201.12086) | [![GitHub](https://img.shields.io/github/stars/salesforce/BLIP?style=social)](https://github.com/salesforce/BLIP) | ✔️ | [License](https://github.com/salesforce/BLIP/blob/main/LICENSE.txt) |
| MaskCLIP | 2022 | 8 | MaskCLIP: Masked Self-Distillation Advances Contrastive Language-Image Pretraining | MLM+MSD | [![arXiv](https://img.shields.io/badge/arXiv-2208.12262-b31b1b.svg)](https://arxiv.org/abs/2208.12262) | [![GitHub](https://img.shields.io/github/stars/LightDXY/MaskCLIP?style=social)](https://github.com/LightDXY/MaskCLIP) | ❌ |  |
| ViCHA | 2022 | 8 | Efficient Vision-Language Pretraining with Visual Concepts and Hierarchical Alignment | H-ITC+ITM+MMM+MIM+MLM | [![arXiv](https://img.shields.io/badge/arXiv-2208.13628-b31b1b.svg)](https://arxiv.org/abs/2208.13628) | [![GitHub](https://img.shields.io/github/stars/mshukor/ViCHA?style=social)](https://github.com/mshukor/ViCHA) | ✔️ | [License](https://github.com/mshukor/ViCHA/blob/main/LICENSE) |
| RILS | 2023 | 1 | RILS: Masked Visual Reconstruction in Language Semantic Space | MIM | [![arXiv](https://img.shields.io/badge/arXiv-2301.06958-b31b1b.svg)](https://arxiv.org/abs/2301.06958) | [![GitHub](https://img.shields.io/github/stars/hustvl/RILS?style=social)](https://github.com/hustvl/RILS) | ❌ |  |
| MobileCLIP | 2023 | 11 | MobileCLIP: Fast Image-Text Models through Multi-Modal Reinforced Training | MMR | [![arXiv](https://img.shields.io/badge/arXiv-2311.17049-b31b1b.svg)](https://arxiv.org/abs/2311.17049) |  | ❌ |  |
<!--- CLIP_PLUS_TABLE -->


## Contrastive Pretraining for Other Modalities

This section contains collections of papers that are related to contrastive pretraining for other modalities, such as audio, video, and 3D data.

### Audio

Models that use CLIP-style contrastive learning as a pretraining objective for audio.

<!--- AUDIO_TABLE -->
| **Model** | **Year** | **Month** | **Paper Title** | **Modalities** | **Arxiv** | **Github** | **Open Source** | **License** |
|:---------:|:---------:|:--------:|:----------------:|:----------------:|:--------:|:--------:|:--------:|:--------:|
| AudioCLIP | 2021 | 6 | AudioCLIP: Extending CLIP to Image, Text and Audio | audio+image+text | [![arXiv](https://img.shields.io/badge/arXiv-2106.13043-b31b1b.svg)](https://arxiv.org/abs/2106.13043) | [![GitHub](https://img.shields.io/github/stars/AndreyGuzhov/AudioCLIP?style=social)](https://github.com/AndreyGuzhov/AudioCLIP) | ✔️ | [License](https://github.com/AndreyGuzhov/AudioCLIP/blob/master/LICENSE.md) |
| WAV2CLIP | 2021 | 10 | WAV2CLIP: LEARNING ROBUST AUDIO REPRESENTATIONS FROM CLIP | audio+image+text | [![arXiv](https://img.shields.io/badge/arXiv-2110.11499-b31b1b.svg)](https://arxiv.org/abs/2110.11499) | [![GitHub](https://img.shields.io/github/stars/descriptinc/lyrebird-wav2clip?style=social)](https://github.com/descriptinc/lyrebird-wav2clip) | ✔️ | [License](https://github.com/descriptinc/lyrebird-wav2clip/blob/master/LICENSE.md) |
| SpeechCLIP | 2022 | 10 | SpeechCLIP: Integrating Speech with Pre-Trained Vision and Language Model | speech+image+text | [![arXiv](https://img.shields.io/badge/arXiv-2210.00705-b31b1b.svg)](https://arxiv.org/abs/2210.00705) | [![GitHub](https://img.shields.io/github/stars/atosystem/SpeechCLIP?style=social)](https://github.com/atosystem/SpeechCLIP) | ✔️ | [License](https://github.com/atosystem/SpeechCLIP/blob/main/LICENSE) |
| CLAP | 2023 | 4 | Large-Scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation | audio+text | [![arXiv](https://img.shields.io/badge/arXiv-2211.06687-b31b1b.svg)](https://arxiv.org/abs/2211.06687) | [![GitHub](https://img.shields.io/github/stars/LAION-AI/CLAP?style=social)](https://github.com/LAION-AI/CLAP) | ✔️ | [License](https://github.com/LAION-AI/CLAP/blob/main/LICENSE) |
| CLVP | 2023 | 5 | Better speech synthesis through scaling | speech+text | [![arXiv](https://img.shields.io/badge/arXiv-2305.07243-b31b1b.svg)](https://arxiv.org/abs/2305.07243) | [![GitHub](https://img.shields.io/github/stars/neonbjb/tortoise-tts?style=social)](https://github.com/neonbjb/tortoise-tts) | ✔️ | [License](https://github.com/neonbjb/tortoise-tts/blob/main/LICENSE) |
<!--- AUDIO_TABLE -->

### Video

Models that extend CLIP to the video domain.

<!--- VIDEO_TABLE -->
| **Model** | **Year** | **Month** | **Paper Title** | **Arxiv** | **Github** | **Open Source** | **License** |
|:---------:|:---------:|:--------:|:----------------:|:--------:|:--------:|:--------:|:--------:|
| CLIP4Clip | 2021 | 4 | CLIP4Clip: An Empirical Study of CLIP for End to End Video Clip Retrieval | [![arXiv](https://img.shields.io/badge/arXiv-2207.07285-b31b1b.svg)](https://arxiv.org/abs/2207.07285) | [![GitHub](https://img.shields.io/github/stars/ArrowLuo/CLIP4Clip?style=social)](https://github.com/ArrowLuo/CLIP4Clip) | ✔️ | [License](https://github.com/ArrowLuo/CLIP4Clip/blob/master/LICENSE) |
| VideoCLIP | 2021 | 9 | VideoCLIP: Contrastive Pre-training for Zero-shot Video-Text Understanding | [![arXiv](https://img.shields.io/badge/arXiv-2109.14084-b31b1b.svg)](https://arxiv.org/abs/2109.14084) | [![GitHub](https://img.shields.io/github/stars/facebookresearch/fairseq?style=social)](https://github.com/facebookresearch/fairseq/tree/main/examples/MMPT) | ✔️ | [License](https://github.com/facebookresearch/fairseq/tree/main/examples/MMPT#copyright) |
| X-CLIP | 2022 | 7 | X-CLIP: End-to-End Multi-grained Contrastive Learning for Video-Text Retrieval | [![arXiv](https://img.shields.io/badge/arXiv-2207.07285-b31b1b.svg)](https://arxiv.org/abs/2207.07285) | [![GitHub](https://img.shields.io/github/stars/xuguohai/X-CLIP?style=social)](https://github.com/xuguohai/X-CLIP) | ✔️ | [License](https://github.com/xuguohai/X-CLIP/blob/main/LICENSE) |
<!--- VIDEO_TABLE -->

### 3D

Models that extend CLIP to the 3D domain.

<!--- 3D_TABLE -->
| **Model** | **Year** | **Month** | **Paper Title** | **Modalities** | **Arxiv** | **Github** | **Open Source** | **License** |
|:---------:|:---------:|:--------:|:----------------:|:--------:|:--------:|:--------:|:--------:|:--------:|
| PointCLIP | 2021 | 12 | PointCLIP: Point Cloud Understanding by CLIP | point cloud + text | [![arXiv](https://img.shields.io/badge/arXiv-2112.02413-b31b1b.svg)](https://arxiv.org/abs/2112.02413) | [![GitHub](https://img.shields.io/github/stars/ZrrSkywalker/PointCLIP?style=social)](https://github.com/ZrrSkywalker/PointCLIP) | ✔️ |  |
| CLIP2Point | 2022 | 10 | CLIP2Point: Transfer CLIP to Point Cloud Classification with Image-Depth Pre-training | point cloud + text | [![arXiv](https://img.shields.io/badge/arXiv-2210.01055-b31b1b.svg)](https://arxiv.org/abs/2210.01055) | [![GitHub](https://img.shields.io/github/stars/tyhuang0428/CLIP2Point?style=social)](https://github.com/tyhuang0428/CLIP2Point) | ✔️ |  |
| PointCLIPV2 | 2022 | 11 | PointCLIP V2: Prompting CLIP and GPT for Powerful 3D Open-world Learning | point cloud + text | [![arXiv](https://img.shields.io/badge/arXiv-2211.11682-b31b1b.svg)](https://arxiv.org/abs/2211.11682) | [![GitHub](https://img.shields.io/github/stars/PointCLIP_V2?style=social)](https://github.com/PointCLIP_V2) | ❌ |  |
| CLIP2 | 2023 | 3 | CLIP2: Contrastive Language-Image-Point Pretraining from Real-World Point Cloud Data | point cloud + image + text | [![arXiv](https://img.shields.io/badge/arXiv-2303.12417-b31b1b.svg)](https://arxiv.org/abs/2303.12417) |  | ❌ |  |
<!--- 3D_TABLE -->


## 👋 Contributing

Contributions are welcome! Submit a pull request to add a new paper, or to update an existing paper. Please follow the format of the existing papers in the table 😄


