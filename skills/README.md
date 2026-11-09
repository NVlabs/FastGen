# FastGen agent skills

[Agent Skills](https://agentskills.io/specification) that teach coding agents such as Claude
Code and Codex how to work with [NVIDIA FastGen](https://docs.nvidia.com/fastgen).

## Install

FastGen skills are published in the [NVIDIA skills catalog](https://github.com/NVIDIA/skills#skill-catalog).
Follow [NVIDIA's installation guide](https://github.com/NVIDIA/skills#quickstart), or, if
NVIDIA Skill Finder is already available in your agent, ask it to install the FastGen skills.

Alternatively, copy a complete skill directory, including its supporting files, into
your agent's skills folder.

## Skills

| Skill | Use it when |
|-------|-------------|
| [fastgen-develop](fastgen-develop/SKILL.md) | FastGen code needs an implementation, refactor or review following shared design, documentation and test conventions. |
| [fastgen-setup](fastgen-setup/SKILL.md) | The environment, credentials, weights or datasets are not yet known to work. |
| [fastgen-runtime](fastgen-runtime/SKILL.md) | A command needs its checkout, interpreter, config and override syntax, roots or launch variables resolved. |
| [fastgen-configure](fastgen-configure/SKILL.md) | A run needs a new or modified experiment config, or a distillation method has to be chosen. |
| [fastgen-dataset](fastgen-dataset/SKILL.md) | Training data is missing, malformed or needs a new data config. |
| [fastgen-train](fastgen-train/SKILL.md) | Training is executed, resumed or misbehaves. |
| [fastgen-inference](fastgen-inference/SKILL.md) | Teacher or student outputs are generated with the inference scripts or ComfyUI nodes. |
| [fastgen-validate](fastgen-validate/SKILL.md) | A claim about correctness, parity or quality needs evidence. |
| [fastgen-add-method](fastgen-add-method/SKILL.md) | An algorithm is not expressible through the existing method configs. |
| [fastgen-add-network](fastgen-add-network/SKILL.md) | A model family, checkpoint or task variant is not yet supported. |
| [fastgen-add-modality](fastgen-add-modality/SKILL.md) | A new signal type must cross the data, model and output interfaces. |

See [Contributing Skills](../CONTRIBUTING.md#contributing-skills) for authoring and review.

## Evaluation

Each skill's `evals/` exercises FastGen code or workflows: source patches with regression
tests, experiment configs, loader artifacts, launch commands and source-backed diagnosis.
Fixtures identify the released source revision; deliberately broken candidates and
authored observations are labeled. Adjacent FastGen requests test routing away from the
target skill and do not count as positive workflow coverage.

The default tasks need no weights, credentials or GPUs. They execute available artifact
checks and require genuine FastGen imports in proposed component tests. A source-only
trial must report those tests as unrun until executed in a complete environment; neither
syntax checks nor evaluator scores establish model parity, trainability or quality.

With [SkillEvaluator](https://docs.nvidia.com/skills/skillevaluator/installation) installed,
check a taskset before running it (repeat for each skill):

```bash
skillevaluator tier3 validate skills/fastgen-inference --strict
skillevaluator tier3 evaluate skills/fastgen-inference --agents codex --env-mode docker
```

The workspace config includes companion skills; the baseline omits only the target skill.
Each case selects its own fixtures.
These configs target CI's SkillEvaluator 1.6.7, which resolves included paths relative
to the skill directory. Recheck that rule when upgrading the evaluator.
Review both comparison arms and the produced artifacts. Keep generated reports outside
the repository with `--results-dir`, and retain model-scale validation as separate evidence.
