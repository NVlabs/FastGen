# FastGen agent skills

[Agent Skills](https://agentskills.io/specification) that teach coding agents such as Claude
Code and Codex how to work with NVIDIA FastGen. Each skill is a directory with a `SKILL.md`
and, where a task has a contract to satisfy, a `references/` folder that states it. The
skills link to the READMEs and docstrings that own the detail instead of restating it, stay
with the general API and design choices rather than specific culprits, and pin no revisions.

## Install

Use the [`skills` CLI](https://github.com/vercel-labs/skills), the same flow as for the
[NVIDIA skills catalog](https://github.com/NVIDIA/skills):

```bash
npx skills add NVlabs/FastGen --list                                   # browse
npx skills add NVlabs/FastGen                                          # install interactively
npx skills add NVlabs/FastGen --skill fastgen-train -a claude-code -y  # one skill, one agent
```

Alternatively, copy a skill directory into your agent's skills folder, such as
`.claude/skills/`.

## Skills

| Skill | Use it when |
|-------|-------------|
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

## Contributing

Keep a `SKILL.md` near 50 lines and put contracts, design choices and general pitfalls into
`references/`. Link to the owning README or docstring with an absolute
`https://github.com/NVlabs/FastGen/blob/main/...` URL so that a copied skill still resolves;
only a skill's own `references/` files are linked relatively. Verify every link and heading
anchor. Pull requests that touch `skills/` must carry the NVSkills CI status, which a
maintainer requests by commenting `/nvskills-ci` on the pull request after each push. See
[CONTRIBUTING.md](../CONTRIBUTING.md) for the general workflow.
