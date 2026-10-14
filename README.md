# DeepSeek-V4.1-Flash inference code, carried for vXPU

The `inference/` and `encoding/` directories, `config.json` and `LICENSE`
are copied verbatim from the Hugging Face repository
`deepseek-ai/DeepSeek-V4.1-Flash` at commit
`5254448103b6254fc646dacbca1cbbf20e9216c1` (MIT license, Copyright (c)
2023 DeepSeek). The weights are not here; vXPU binds them from the
checkpoint by content hash.

The `vxpu` branch adds what `torch.export` needs to capture this model
weightlessly on a machine without a GPU: plain-aten versions of the
tilelang kernels (fp8/fp4 quantization and matmuls, sparse attention
with an attention sink, the hyper-connection sinkhorn split), static
cache state instead of Python-side buffers, and a static-shape
formulation of the index selection. Each change is kept small and
documented so it can be read against the original. See
github.com/gke-labs/generation-ai, `vxpu/third_party/`.
