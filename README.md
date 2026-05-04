## GPU Support
XAIDissect will be a GPU visual accelerator through CUDA.jl with the option to use CPU later.
This repo does not vendor or redistribute the NVIDIA CUDA Toolkit, drivers, model weights or properietary NVIDIA libraries.  Users are responsible for installing compatible NVIDIA drivers and accepting any upstream NVIDIA/CUDA licenses required by their system.

## Purpose
Visual representation of open weights xAI/Grok-1 MoE checkpoints using exported 'xai-dissect' JSON reports.  This all includes cartography: blocks, routers, experts, routing-critical tensors, and SAAQ-readiness metrics.

This repo is not a model runner nor does it redistribute model weights.

## Planned Features

- Load 'xai-dissect' JSON reports
- Visualize the routers, experts and blocks
- Rendor router risk and readiness heatmaps
- Simulate router logits and tok-k expert selection
- Support CPU execution at first
- Main goal: NVIDIA accelerator through CUDA.jl

## Relationship to xai-dissect

- 'xai-dissect' extracts checkpoint structure and emits reports.
- 'XAIDissectViz.jl visualizes the reports interactively.
