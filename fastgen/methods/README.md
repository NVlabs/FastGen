# FastGen Methods

Training methods for fast single-step or few-step generation from diffusion models.

## Methods Overview

| Category | README | Method | Class | Description | Reference |
|----------|--------|--------|-------|-------------|-----------|
| **Consistency** | [README](consistency_model/README.md) | CM | [`CMModel`](consistency_model/CM.py) | Consistency model (tuning/distillation) | [Song et al., 2023](https://arxiv.org/abs/2303.01469), [Geng et al., 2024](https://arxiv.org/abs/2406.14548) |
| | | sCM | [`SCMModel`](consistency_model/sCM.py) | Continuous-time CM with TrigFlow | [Lu & Song, 2024](https://arxiv.org/abs/2410.11081) |
| | | TCM | [`TCMModel`](consistency_model/TCM.py) | Two-stage CM with boundary loss | [Lee et al., 2024](https://arxiv.org/abs/2410.14895) |
| | | MeanFlow | [`MeanFlowModel`](consistency_model/mean_flow.py) | Mean velocity prediction | [Geng et al., 2025](https://arxiv.org/abs/2505.13447), [Sabour et al., 2025](https://arxiv.org/abs/2506.14603) |
| **Distribution Matching** | [README](distribution_matching/README.md) | DMD2 | [`DMD2Model`](distribution_matching/dmd2.py) | VSD + GAN distillation | [Yin et al., 2024](https://arxiv.org/abs/2405.14867) |
| | | f-distill | [`FdistillModel`](distribution_matching/f_distill.py) | f-divergence weighted DMD2 | [Xu et al., 2025](https://arxiv.org/abs/2502.15681) |
| | | LADD | [`LADDModel`](distribution_matching/ladd.py) | Pure adversarial distillation | [Sauer et al., 2024](https://arxiv.org/abs/2403.12015) |
| | | CausVid | [`CausVidModel`](distribution_matching/causvid.py) | Causal DMD2 with diffusion forcing | [Yin et al., 2024](https://arxiv.org/abs/2412.07772) |
| | | AnyFlow | [`AnyFlowModel`](distribution_matching/anyflow.py) | Any-step DMD2 with a flow-map student | [Gu et al., 2026](https://arxiv.org/abs/2605.13724) |
| | | Self-Forcing | [`SelfForcingModel`](distribution_matching/self_forcing.py) | Causal DMD2 with self-forcing | [Huang et al., 2025](https://arxiv.org/abs/2506.08009) |
| **Fine-Tuning** | [README](fine_tuning/README.md) | SFT | [`SFTModel`](fine_tuning/sft.py) | Finetuning with denoising score matching | [Ho et al., 2020](https://arxiv.org/abs/2006.11239), [Song et al., 2020](https://arxiv.org/abs/2011.13456), [Lipman et al., 2022](https://arxiv.org/abs/2210.02747), [Albergo et al., 2023](https://arxiv.org/abs/2303.08797) |
| | | CausalSFT | [`CausalSFTModel`](fine_tuning/sft.py) | Causal version of SFT | [Chen et al., 2024](https://arxiv.org/abs/2407.01392) |
| **Knowledge Distillation** | [README](knowledge_distillation/README.md) | KD | [`KDModel`](knowledge_distillation/KD.py) | Learn from pre-computed trajectories | [Luhman & Luhman, 2021](https://arxiv.org/abs/2101.02388) |
| | | CausalKD | [`CausalKDModel`](knowledge_distillation/KD.py) | Causal version of KD | [Yin et al., 2024](https://arxiv.org/abs/2412.07772) |


