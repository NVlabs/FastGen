# FastGen Methods

Training methods for fast single-step or few-step generation from diffusion models.

**Multimodal distillation:** DMD2, PDD, and SFT natively support multimodal data (e.g., joint video+audio). When the input is a dict of tensors (e.g., `{"video": ..., "audio": ...}`), losses are computed independently per modality and summed automatically. Per-modality CFG scales can be specified as a dict (e.g., `guidance_cfg.cfg.scale={"video": 3.0, "audio": 7.0}`).

## Methods Overview

| Category | README | Method | Class | Description | Reference |
|----------|--------|--------|-------|-------------|-----------|
| **Trajectory Matching** | [README](trajectory_matching/README.md) | CM | [`CMModel`](trajectory_matching/CM.py) | Consistency model (tuning/distillation) | [Song et al. (2023)](https://arxiv.org/abs/2303.01469), [Geng et al. (2024)](https://arxiv.org/abs/2406.14548) |
| | | sCM | [`SCMModel`](trajectory_matching/sCM.py) | Continuous-time CM with TrigFlow | [Lu & Song (2024)](https://arxiv.org/abs/2410.11081) |
| | | TCM | [`TCMModel`](trajectory_matching/TCM.py) | Two-stage CM with boundary loss | [Lee et al. (2024)](https://arxiv.org/abs/2410.14895) |
| | | MeanFlow | [`MeanFlowModel`](trajectory_matching/mean_flow.py) | Mean velocity prediction | [Geng et al. (2025)](https://arxiv.org/abs/2505.13447), [Sabour et al. (2025)](https://arxiv.org/abs/2506.14603) |
| | | AYF | [`AYFModel`](trajectory_matching/ayf.py) | Flow-map distillation with the EMD objective | [Sabour et al. (2025)](https://arxiv.org/abs/2506.14603) |
| | | KD | [`KDModel`](trajectory_matching/KD.py) | Learn from pre-computed trajectories | [Luhman & Luhman (2021)](https://arxiv.org/abs/2101.02388) |
| | | CausalKD | [`CausalKDModel`](trajectory_matching/KD.py) | Causal version of KD | [Yin et al. (2024)](https://arxiv.org/abs/2412.07772), [Zhu et al. (2026)](https://arxiv.org/abs/2602.02214) |
| | | PDD | [`PDDModel`](trajectory_matching/pdd.py) | Predict multiple trajectory velocities per network evaluation | [Shaul et al. (2026)](https://arxiv.org/abs/2607.26004) |
| **Distribution Matching** | [README](distribution_matching/README.md) | DMD2 | [`DMD2Model`](distribution_matching/dmd2.py) | VSD + GAN distillation | [Yin et al. (2024)](https://arxiv.org/abs/2405.14867) |
| | | f-distill | [`FdistillModel`](distribution_matching/f_distill.py) | f-divergence weighted DMD2 | [Xu et al. (2025)](https://arxiv.org/abs/2502.15681) |
| | | LADD | [`LADDModel`](distribution_matching/ladd.py) | Pure adversarial distillation | [Sauer et al. (2024)](https://arxiv.org/abs/2403.12015) |
| | | CausVid | [`CausVidModel`](distribution_matching/causvid.py) | Causal DMD2 with diffusion forcing | [Yin et al. (2024)](https://arxiv.org/abs/2412.07772) |
| | | AnyFlow | [`AnyFlowModel`](distribution_matching/anyflow.py) | Any-step DMD2 with a flow-map student | [Gu et al. (2026)](https://arxiv.org/abs/2605.13724) |
| | | Self-Forcing | [`SelfForcingModel`](distribution_matching/self_forcing.py) | Causal DMD2 with self-forcing | [Huang et al. (2025)](https://arxiv.org/abs/2506.08009) |
| **Fine-Tuning** | [README](fine_tuning/README.md) | SFT | [`SFTModel`](fine_tuning/sft.py) | Finetuning with denoising score matching | [Ho et al. (2020)](https://arxiv.org/abs/2006.11239), [Song et al. (2020)](https://arxiv.org/abs/2011.13456), [Lipman et al. (2022)](https://arxiv.org/abs/2210.02747), [Albergo et al. (2023)](https://arxiv.org/abs/2303.08797) |
| | | CausalSFT | [`CausalSFTModel`](fine_tuning/sft.py), [`CausalTeacherForcingSFTModel`](fine_tuning/sft.py) | Causal finetuning with diffusion/teacher forcing | [Yin et al. (2024)](https://arxiv.org/abs/2412.07772), [Zhu et al. (2026)](https://arxiv.org/abs/2602.02214) |
