# Corrected multivariate head sweep checklist

- [x] Correct extractor uses TimesFM3 variate attention and target masks for observed channels.
- [x] Corrected UCR and UEA array jobs submitted on H100/A100 GPUs.
- [x] Balanced accuracy added to generic evaluation output.
- [x] Repartition route superseded: training now reads completed UCR/UEA roots directly and uses 100% of eligible source examples.
- [x] FlaMinGo 27-dataset evaluation-only manifest created; those datasets excluded from architecture-selection training.
- [x] Shared multi-dataset evaluation path added with stable train+test label maps.
- [x] Attention head bounded for high-channel UEA inputs.
- [x] Extractor bounded both inference batch size and stored high-channel representation.
- [x] Recovery array resubmitted for all UEA shards with the bounded cache policy.
- [ ] Verify all 158 corrected dataset feature pairs.
- [x] Deferred UEA manifest created for on-the-fly extraction.
- [ ] Launch four parallel head variants, each within two hours.
- [ ] Compare held-out OOD metrics against existing baseline and FlaMinGo.
- [ ] Run stratified 3/5-fold PS3 evaluation and fine-tuning.
- [ ] Update progress report with exact artifacts and limitations.
