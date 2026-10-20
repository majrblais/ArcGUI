# ArcGUI Model Weights

The ONNX model weights are distributed separately and are not stored in the public GitHub repository.

## Access

Contact:

- **Laurie Vidal**
- **Dodick Gasser**

The project weights are located in the **CCNB Innov – Chemin forestier** OneDrive shared by Laurie Vidal under the **`final`** folder.

## Recommended Models

### Primary final model

```text
segformer_cuda_fp16.onnx
```

### Faster CUDA alternative

```text
best_unet_resnet101_norm_1024_cuda_fp16.onnx
```

## Additional Distributed Models

```text
segformer_cpu_fp32_optimized.onnx
segformer_last_cuda_fp16.onnx
last_unet_resnet101_norm_1024_cuda_fp16.onnx
old_milestone_unet_resnet101_norm_1024.onnx
```

These are mainly retained for CPU-only use, comparison, reproducibility, or compatibility.

## Important

Keep the distributed filenames unless there is a specific reason to rename them.

ArcGUI uses model-family/provider keywords in the filename to select the expected execution behavior.

Before first deployment on a new computer, run:

```bash
python check.py
```

and select the desired model.
