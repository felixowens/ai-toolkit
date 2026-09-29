# Paired concept training on Krea2

The two example configs run separate ass-size experiments on the same paired
manifest. `train_paired_concept_krea2.example.json` uses a trigger word
(`asssize`). `train_paired_slider_krea2.example.json` uses signed adapter
strength: `-1` for the smaller end, `0` for the base model, and `+1` for the
larger end. They are separate runs with separate output names.

For the initial three-pair experiment, `ass_size_pairs.review.json` contains
the captions and VM image paths. The images and a copy of the manifest are
staged at `/home/Ubuntu/workspace/datasets/ass_size_pairs/` on the VM; image
files are not part of the GitHub branch.

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
must contain the trigger; the negative caption must omit it. For the slider
run, the loader uses the negative caption for both images, so conditioning is
identical at both ends. The dataset loader keeps pairs together through
batching, latent caching, and text embedding caching.
Cropping is deterministic and shared by both images. Random crop, scale, and
flips are disabled for this first implementation.

Both training steps use the same noise and timestep for both latents. The
trigger run has positive flow reconstruction, frozen-base preservation of the
negative image, and a difference loss. The slider run reconstructs both images
at `+1` and `-1` strength, matches their prediction difference to the target
difference, and penalizes average deviation from the frozen model to keep the
two directions centered.
The difference loss is weighted toward high-noise timesteps, where the two noisy
latents converge and the caption must carry the concept distinction. No masks
are used. `paired_difference_multiplier`, `paired_preservation_multiplier`, and
`paired_difference_timestep_power` control the objective.

The example samples compare base, trigger-on, and trigger-off for the concept
run, and `-1`, `0`, and `+1` for the slider run. This implementation is currently
scoped to Krea2 flow matching, frozen text encoder, and ordinary LoRA/LoKr
networks.
