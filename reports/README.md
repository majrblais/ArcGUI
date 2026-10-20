# ArcGUI `reports/`

This folder contains the complete project documentation.

The root `README.md` is intended as a quick-start deployment reference. For full details, use the PDFs in this folder.

---

## `technical_guide.pdf`

The **technical deployment guide** is the main user guide for ArcGUI.

It covers:

- environment setup;
- ONNX / CUDA verification;
- model selection;
- ArcGIS toolbox installation;
- GUI parameters;
- coordinate-system requirements;
- direct and optimized export modes;
- 1024 × 1024 overlap behavior;
- reusable imagery caching;
- debug footprint preview;
- probability reconstruction;
- thresholding;
- output structure;
- safeguards and troubleshooting;
- recommended operational workflow.

---

## `technical_report.pdf`

The **technical report** documents the research and model-development work behind ArcGUI.

It includes:

- dataset preparation;
- annotation issues and corrections;
- training methodology;
- architecture/loss comparisons;
- U-Net and SegFormer evaluation;
- held-out validation;
- qualitative examples;
- threshold analysis;
- runtime benchmarks;
- final model selection;
- conclusions;
- future work.

The final project model is SegFormer MiT-B5, while the best U-Net ResNet101 model is retained as a faster deployment alternative.

---

# Recommended Deployment Models

Primary:

```text
segformer_cuda_fp16.onnx
```

Faster CUDA alternative:

```text
best_unet_resnet101_norm_1024_cuda_fp16.onnx
```

Other CPU, last-checkpoint, and old-milestone models are retained mainly for compatibility, comparison, or reproducibility.

---

# Model Access

The ONNX weights are distributed separately from GitHub.

Contact **Laurie Vidal** or **Dodick Gasser**.

The model files are available in the **CCNB Innov – Chemin forestier** OneDrive shared by Laurie Vidal under the **`final`** folder.

---

# Keeping Documentation Current

When the GUI or backend is updated, verify that the deployment guide still reflects:

- current GUI parameters;
- supported model filenames;
- exact-polygon processing behavior;
- recommended batch sizes;
- cache behavior;
- output-folder organization;
- environment requirements.

The technical guide is intended to remain the authoritative detailed deployment document.
