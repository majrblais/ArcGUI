# ArcGUI Model Weights

The ONNX model weights are distributed separately and are not stored in the public GitHub repository.

## Access

Contact:

- **Laurie Vidal**
- **Dodick Gasser**

The project weights are located in the **CCNB Innov – Chemin forestier** OneDrive shared by Laurie Vidal under the **`final`** folder.

---

## Recommended models

Primary final CUDA model:

```text
segformer_cuda_fp16.onnx
```

Faster CUDA alternative:

```text
best_unet_resnet101_norm_1024_cuda_fp16.onnx
```

Additional distributed variants may include:

```text
segformer_cpu_fp32_optimized.onnx
segformer_last_cuda_fp16.onnx
last_unet_resnet101_norm_1024_cuda_fp16.onnx
old_milestone_unet_resnet101_norm_1024.onnx
```

These additional files are mainly retained for CPU-only deployment, compatibility, comparison, or reproducibility.

Before first use on a new computer, run:

```bash
python utils/check.py
```

and select the model you intend to deploy.
