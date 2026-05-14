# Guitar Effector Classifier - BEATs Multi-Axis Inference

This repository contains the minimal files needed to run the trained
`16_beats_multiaxis_intensity` model on WAV files.

## What The Model Predicts

The model receives a WAV file and predicts intensity for three effect axes:

| Axis | Output labels |
| --- | --- |
| `drive` | `off`, `25`, `50`, `75`, `100` |
| `phase` | `off`, `25`, `50`, `75`, `100` |
| `space` | `off`, `25`, `50`, `75`, `100` |

The trained checkpoint is:

```text
16_beats_multiaxis_intensity/artifacts/best_model.pt
```

This file is tracked with Git LFS because it is larger than normal GitHub file
limits.

## Install

```powershell
python -m pip install -r requirements.txt
```

For CUDA acceleration, install a PyTorch build that matches your GPU and CUDA
environment if the default install does not use GPU.

## Run Folder Inference

From the repository root:

```powershell
cd 16_beats_multiaxis_intensity
python infer_folder.py --input-dir "C:\path\to\wav_folder" --output-dir "outputs\my_run"
```

The CSV result will be saved to:

```text
16_beats_multiaxis_intensity/outputs/my_run/single_folder_inference.csv
```

## Run Single-File Inference

```powershell
cd 16_beats_multiaxis_intensity
python infer.py --audio-path "C:\path\to\file.wav"
```

## Important Notes

- The checkpoint was trained mostly on single-effect data.
- External tests showed the model tends to over-predict the `space` axis and
  under-detect the `phase` axis on unseen test folders.
- For reliable multi-effect detection, include multi-effect examples in
  training and fine-tune again.

