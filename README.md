# 🎧 Progressive Homeostatic and Plastic Prompt Tuning for Audio-Visual Multi-Task Incremental Learning (ICCV 2025)

Code for our ICCV 2025 paper on multi-task audio-visual continual learning.  
Paper: https://arxiv.org/pdf/2507.21588

## 📝 Overview
- ⚖️ PHP: Progressive Homeostatic & Plastic prompts (Shallow/Mid/Deep) decoupling stability (memory) vs. plasticity (adaptation).
- 🧠 Modules: Task-shared Aggregation (Shallow) → Dynamic Generating Adapters (Mid) → Modality-independent Prompts (Deep).
- 🚀 Scope: Audio-Visual Multi-Task Incremental Learning (AV-MTIL) across AVE, AVVP, AVS, and AVQA benchmarks.

## 🗂 Repo Map
- `configs/`: experiment JSONs (e.g., `tri_ave_avqa_llp.json`; ablations in subfolders).
- `methods/`: model, prompts, losses, CLIP/HTSAT backbones.
- `data.py`, `data_manager.py`: dataset loading and incremental splits.
- `utils/`: helpers; `script/`: domain/category utilities.
- Logs/checkpoints saved to `logs/<prefix>_*`.

## ⚙️ Environment
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
# optional: disable W&B cloud
set WANDB_MODE=online/offline
```

## Pretrained Weights
Download 630k-audioset-fusion-best.pt and place it under pretrain/models/.

## 📦 Data Preparation
We follow the dataset setup from DG-SCT (https://github.com/haoyi-duan/DG-SCT). Datasets are available via our Baidu Netdisk or Google Drive mirrors; download them and update the roots in `data.py` to your local paths. (The dataset will be upload in two days.)

## 🚀 Quickstart (Train)
Task-incremental example (AVE -> LLP -> AVQA):
```bash
python main.py --config configs/ave_llp_avqa.json
```

Key config knobs (see the JSON):
- `prefix`: run name (log/checkpoint path)
- `task_order`: e.g., `["AVE", "AVQA", "LLP"]`
- Prompt settings: `use_shallow_adapter`, `use_mid_prompt`, `use_deep_prompt`, `prompt_length`, etc.
- Optim: `init_lr`, `epochs`, `batch_size`, `num_workers`

## 🙏 Acknowledgements
Thanks to the DG-SCT project (https://github.com/haoyi-duan/DG-SCT) for the dataset setup inspiration and resources.

## 📚 Citation
If this repo helps your research, please cite:
```bibtex
@inproceedings{yin2025progressive,
  title={Progressive Homeostatic and Plastic Prompt Tuning for Audio-Visual Multi-Task Incremental Learning},
  author={Yin, Jiong and Li, Liang and Zhang, Jiehua and Gao, Yuhan and Yan, Chenggang and Sheng, Xichun},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={2022--2033},
  year={2025}
}
```