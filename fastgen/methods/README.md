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

## Choosing a distillation baseline

These are practical heuristics; validate quality and diversity for the chosen task. Starting with **8 NFE** is a
good rule of thumb.

- **[DMD2](distribution_matching/README.md#dmd2-distribution-matching-distillation-v2):** distribution matching for
  fast convergence and high visual quality even with 4 NFE or less. Watch for unstable dynamics or diversity loss; retain the
  best checkpoint and consider early stopping when quality regresses.
- **[PDD](trajectory_matching/README.md#pdd-parallel-decoding-distillation):** trajectory matching for teacher
  fidelity, diversity and more stable training, typically at 4–8 NFE.

**Data-free DMD2 and PDD:** use backward simulation instead of noising real data. Start with
`model.backward_simulation="staggered"` and
[`CarryCallback`](../callbacks/carry.py); respect the [rollout phase constraints](backward_simulation.py).
For DMD2, enable VSD and explicitly disable GAN (`model.gan_loss_weight_gen=0`). Both still require a teacher and task
conditioning, including source media or cached conditions for I2I/V2V.

### Causal distillation pipeline

Follow [Causal Forcing](https://arxiv.org/abs/2602.02214), reusing compatible checkpoints where available:

1. **[Teacher-forcing causal SFT](fine_tuning/README.md#causalsft)** adapts the original pretrained model into a causal teacher.
2. **[Causal KD](trajectory_matching/README.md#causalkd)** trains on that teacher's cached trajectories and clean
   targets; align saved timesteps with the student's grid.
3. **[Self-Forcing](distribution_matching/README.md#self-forcing)** initializes the causal student from KD and
   distills against the original pretrained teacher used to initialize stage 1, with `model.backward_simulation="default"`.
