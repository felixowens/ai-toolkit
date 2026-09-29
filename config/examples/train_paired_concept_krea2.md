# Paired concept training on Krea2

`train_paired_concept_krea2.example.json` is a review template for the ass-size
experiment. It is not launchable until the paired dataset exists. It produces a
triggered concept adapter; the example trigger is `asssize`.

The dataset manifest is a JSON object keyed by absolute positive image paths:

```json
{
  "/path/large.png": {
    "caption": "A woman in leggings with asssize, viewed from behind",
    "negative_image": "/path/small.png",
    "negative_caption": "A woman in leggings, viewed from behind"
  }
}
```

Both images must have identical dimensions and alignment. Captions should share
their scene wording and differ only in the concept phrase. The positive caption
must contain the trigger; the negative caption must omit it. The dataset loader
keeps pairs together through batching, latent caching, and text embedding caching.
Cropping is deterministic and shared by both images. Random crop, scale, and
flips are disabled for this first implementation.

The training step uses the same noise and timestep for both latents. Its three
losses are ordinary flow reconstruction of the positive image, frozen-base
preservation of the negative image, and the difference between positive and
negative flow predictions against the difference between their flow targets.
The difference loss is weighted toward high-noise timesteps, where the two noisy
latents converge and the caption must carry the concept distinction. No masks
are used. `paired_difference_multiplier`, `paired_preservation_multiplier`, and
`paired_difference_timestep_power` control the objective.

The example samples compare the base output, the adapter without the trigger,
and the adapter with the trigger. This implementation is currently scoped to
Krea2 flow matching, frozen text encoder, and ordinary LoRA/LoKr networks.
