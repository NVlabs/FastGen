# Custom training integration

Use FastGen losses in your own training loop with your models, optimizers,
and checkpoints. FastGen's trainer uses the default unmasked losses and
iteration-based phase selection. Supply weights, select phases, or manage
ratio statistics explicitly when your loop needs that control.

## Mask and weight losses

`variational_score_distillation_loss`, `denoising_score_matching_loss`, and
`gan_loss_generator` accept a keyword-only `loss_weight`. The discriminator loss
accepts `real_weight` and `fake_weight`. Weights must be finite, nonnegative, and
broadcast-compatible with the predictions or discriminator scores. Positive
weights scale squared residuals or score losses linearly. Zero weights exclude
entries from the loss and from VSD target normalization. Means divide by the
full element count, including masked entries. Omit
weights to include every entry with unit weight.

```python
loss = variational_score_distillation_loss(
    generated, teacher_x0, fake_x0, loss_weight=pixel_weight,
)
```

The sCM and MeanFlow `_scm_pred_to_loss` and `_mf_pred_to_loss` extension points
accept the same keyword to exclude masked entries from tangent norms and
adaptive weights. Samples with no valid entries contribute zero loss,
including the sCM log-variance term.

MeanFlow clips `t - r` to the noise scheduler's time bounds by default. Set
`clip_time_interval=False` when `t - r` already expresses the duration required
by your custom time parameterization. For DSM flow prediction, supply
`flow_target` to replace the default velocity `eps - x0` with a conditional
velocity from your custom path.

For adversarial losses, exclude corrupted inputs during feature extraction
and pooling, then supply weights for the resulting discriminator scores.
A pixel mask alone cannot identify valid scores. Masking final scores cannot
undo corruption that has already spread through the backbone.

## Phase selection

```python
losses, outputs = model.single_train_step(batch, iteration, phase="auxiliary")
```

DMD2 and f-distill accept `phase="student"` or `phase="auxiliary"`
independently of iteration. Pass the actual training step as `iteration` for
warmup schedules, and step the optimizer matching your selected phase. Omit
`phase` to follow `student_update_freq`, as FastGen's trainer does.

`single_train_step` prepares inputs, samples noise, and sets model modes and
gradients for the selected phase. Accept the `phase` keyword when overriding
noise generation in a subclass. Use `_dispatch_training_phase` with prepared
inputs when your loop also manages sampling, model modes, and frozen parameters.

## Ratio state and process groups

Use `f_divergence_weights` to compute f-distill weights while keeping running
ratio statistics in your own training state:

```python
h, new_bins = f_divergence_weights(
    scores, t, weighting_fn, bins=bins,
    min_t=min_t, max_t=max_t,
    logit_weight=score_weight,
    statistics_group=data_group,
    update_state=True,
)
```

Save the returned bins for the next call or in a checkpoint. The function does
not change the input bins. Set `update_state=False` to normalize with saved
statistics without updating them. Omit `bins` to skip bin normalization.
Zero `logit_weight` entries exclude scores from ratio statistics. Positive
magnitudes only mark scores as valid. Use `sample_mask` to exclude whole
samples. Samples with no valid scores receive zero weight.

Call the function on every rank in `statistics_group`, even if a rank has no
valid samples. This group should join data replicas, not spatial partitions
of one sample. Combine spatial features within your model when needed. During
distributed execution, omitting `statistics_group` uses the
default process group.

Normalization uses the actual global valid count, supporting unequal batch
sizes. Scalar divergence functions broadcast to the batch, including reverse KL.
FastGen's f-distill method uses this function and stores bins on its student
network when you enable ratio normalization. Its checkpoint format stays
unchanged. External loops can instead checkpoint the returned bins.
