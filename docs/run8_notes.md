# Run 8: Latent Translation via Denoising Autoencoders

## Motivation

Run 7 (DiT diffusion) failed because independent per-position token prediction can't coordinate structural decisions. The pointwise discretization bottleneck (CoDAR, Shen et al. 2026) has an irreducible optimality gap between continuous latents and discrete tokens.

Run 8 takes a different approach: decouple the translation task from the tokenization task. Train two autoencoders — one for JSON, one for XML — that compress token sequences into continuous latents and reconstruct them near-losslessly. Then train a translator that maps JSON latents to XML latents in the continuous space, where no discretization happens mid-pipeline. Only the final XML decoder discretizes back to tokens.

The pipeline at inference:

```
JSON tokens → JSON encoder → latent → translator → latent → XML decoder → XML tokens
```

Only the JSON encoder and XML decoder are kept. The JSON decoder and XML encoder are training scaffolding, discarded after the AEs are trained.

## Architecture (final)

Each AE is a hybrid CNN + Transformer:

```
FactoredEmbedding (16k vocab, rank 128, d_emb=384)
    ↓
3× Conv1d (stride 2, kernel 7, channels 384→384→384, GroupNorm + GELU)
    ↓
Transformer stack (384d, 6 heads, d_ff=1152) — N encoder layers
    ↓
[LATENT] — (B, L/8, 384), no projection bottleneck
    ↓
Transformer stack (384d, 6 heads, d_ff=1152) — M decoder layers
    ↓
3× ConvTranspose1d (stride 2, mirrors encoder)
    ↓
LayerNorm + tied vocab projection (via FactoredEmbedding.project_to_vocab)
```

Asymmetric layer counts to match downstream usage:

- **JSON AE**: 4 encoder / 2 decoder layers (~17.2M params). Heavy encoder — it's what we keep.
- **XML AE**: 4 encoder / 4 decoder layers (~20M params). Heavy on both sides — the decoder is kept but needs quality latents to train against, so the encoder can't be weak.

Key design choices:

- **No latent projection** (removed mid-run). The earlier 384→128→384 linear bottleneck was a vestige of the DiT era where diffusion over lower-dimensional latents had better signal-to-noise. For a deterministic transformer translator, the bottleneck just throws away information.
- **Rank-128 FactoredEmbedding** (up from rank-64). 128 degrees of freedom in the embedding matches the 384d pipeline width without being oversized for the 16k vocab.
- **Uniform 384d pipeline**: conv channels, transformer d_model, and latent dim all at 384. Dimension changes lose information; matching widths end-to-end preserves it.
- **8x spatial compression via 3 stride-2 convs**: tested alternative configurations (2 convs at stride 2+4 = 8x) and uniform stride-2 was better. Factored compression preserves more information than aggressive single-step compression.

## Training Recipe (final)

```
./training/run.sh train-ae <json|xml> --freq-weight --noise-frac 0.20 --noise-schedule
```

The recipe in order:

1. **Inverse-sqrt-frequency CE loss weighting** (`--freq-weight`): structured data has extreme token frequency skew (500-1000x between common structural tokens and rare content tokens). Unweighted CE gradient is dominated by common tokens, causing rare tokens to collapse into high-frequency attractors. Weighting by `1/sqrt(freq)` rebalances.
2. **20% token corruption initially** (`--noise-frac 0.20`): acts as regularization during early training to prevent memorization. Random tokens replace 20% of non-pad positions.
3. **Single-shot noise schedule** (`--noise-schedule`): when validation accuracy plateaus, drop noise to 0. Past the memorization risk, denoising becomes a ceiling because the model learns to output "plausible" rare tokens rather than exact ones.
4. **All aug_types included in training**: structurally malformed (`aug_type="corrupted"`), shortened, compacted, truncated, and augmented samples all stay in the training set. The encoder's job is to faithfully encode whatever appears at its input — the downstream translator needs to see a consistent signal regardless of input quality.
5. **ReduceLROnPlateau after noise drops**: first plateau → noise to 0. Subsequent plateaus → LR halved until min_lr=1e-5.

## Results (JSON AE)

Best: **99.98% token accuracy, 0.015% CER, 94.1% perfect reconstruction** (epoch 7, first epoch after noise drop).

Progression:

| Epoch | Noise | Acc    | CER    | Perfect | Note |
|-------|-------|--------|--------|---------|------|
| 1     | 0.20  | 99.16% | 1.01%  | 50.6%   | Bucketed val — long samples already at 99.95% |
| 2     | 0.20  | 99.71% | 0.25%  | 41.3%   |  |
| 6     | 0.20  | 99.84% | 0.16%  | 58.7%   | Plateau at noise=0.20 — denoising ceiling |
| 7     | 0    | 99.98% | 0.015% | 94.1%   | First epoch after noise drop |

The single-shot noise drop gave a 14 basis point accuracy jump and a 10x CER reduction in one epoch. This confirms the noise was the ceiling, not architectural capacity.

## What We Tried That Didn't Work

### Architectural expansions (all gave marginal gains when the real issue was the training objective)

- **Latent expansion 64→128 dim** (`expand_latent.py`): Net2Net widening of the `to_latent` / `from_latent` layers. Small gain (~3pp encoder/decoder fault split shift). The bottleneck wasn't the issue.
- **Transformer depth expansion 2→4 encoder layers** (`edit_layers.py`): identity-initialized layer insertion. ~10% encoder fault reduction. Modest.
- **Conv compression depth** (`edit_convs.py`): reducing from 3 convs at stride 2 to 2 convs at stride 2+4 (same 8x total). Worse — factored compression preserves more information per step.
- **2 conv / 4 dec asymmetric decoder split**: inspired by MAE's heavy-encoder / light-decoder paradigm. Decoder went from 2 to 1 layer, permanent capacity loss. The remaining decoder couldn't give clean training signal, encoder couldn't improve.

### Training signal issues

- **Pure clean + no noise**: immediate memorization. Train loss dropped to 0.0005 in 2 epochs while val diverged. No regularization pressure.
- **High denoising rate maintained through training**: ceiling at ~99.8%. The denoising objective teaches the model to predict plausible tokens, not exact ones. Rare tokens get smeared toward common neighbors.

## Key Findings / Empirical Rules

1. **Denoising is a regularization schedule, not a training objective.** Start with noise to avoid memorization, drop to zero to let rare-token fidelity emerge. 20% for the first few epochs then 0 works.
2. **The encoder's job is faithful encoding.** Whatever the input looks like — well-formed, compacted, truncated, or structurally malformed — the encoder must represent it accurately. Filtering input variations teaches the encoder to only handle a subset of the distribution, which the downstream translator would then have to work around.
3. **Frequency collapse is real and freq-weighting is sufficient to address it.** Inverse-sqrt weighting normalized for mean=1 preserves loss scale while giving rare tokens enough gradient to not collapse.
4. **Information-preserving pipeline width matters.** Any dimension change is a potential information bottleneck. Matching d_emb, conv channels, d_tf, latent_dim = 384 end-to-end removed multiple chokepoints simultaneously.
5. **Asymmetry should match downstream use.** For JSON (encoder kept), heavy encoder + light decoder. For XML (decoder kept), both sides need to be strong because decoder quality depends on encoder quality.
6. **Resume logic must cover three states**: mid-training, mid-validation (with val progress preserved per-chunk), and post-epoch. Missing any of these three leads to lost work on interrupt.

## Related Work / References

### Denoising autoencoders
- **Vincent et al.** *Extracting and Composing Robust Features with Denoising Autoencoders.* ICML 2008. [[paper]](https://www.cs.toronto.edu/~larocheh/publications/icml-2008-denoising-autoencoders.pdf) Original DAE paper. Argued for denoising as a training criterion for robust representations in downstream classification. Our use case (high-fidelity reconstruction) is different from theirs, and we found noise acts as a regularizer — useful early, harmful late.
- **Vincent et al.** *Stacked Denoising Autoencoders.* JMLR 2010. [[paper]](https://www.jmlr.org/papers/volume11/vincent10a/vincent10a.pdf)

### Masked autoencoders / asymmetric designs
- **He, Chen, Xie, Li, Dollár, Girshick.** *Masked Autoencoders Are Scalable Vision Learners (MAE).* CVPR 2022. [[arXiv]](https://arxiv.org/abs/2111.06377) The blueprint for our JSON AE's heavy-encoder / light-decoder design. Key insight: the decoder is throwaway scaffolding if you only want the encoder. MAE uses 1/10 decoder FLOPs vs encoder.
- **Zhu et al.** *Designing a Better Asymmetric VQGAN for StableDiffusion.* arXiv:2306.04632 (2023). Opposite asymmetry — heavier decoder for StableDiffusion. Motivated our XML AE design rationale (keep the decoder, make it strong).
- **LV-RAE.** *Improving Reconstruction of Representation Autoencoder.* arXiv:2602.08620 (2026). 6-layer encoder / 12-layer decoder for image reconstruction.

### Frequency collapse in structured data
- **Hong & Ling.** *Neural Collapse under Cross-Entropy with Imbalanced Data.* 2023. Theoretical foundation: CE loss geometry causes minority class representations to collapse regardless of architecture.
- **Cui et al.** *Class-Balanced Loss Based on Effective Number of Samples.* CVPR 2019. Framework for frequency-dependent loss weighting — our inverse-sqrt weighting is a special case.
- **Jiang et al.** *Why Gender Pronouns Are Difficult: Frequency-Aware Cross-Entropy for Dialogue Generation (FACE).* WWW 2019. Linear inverse-frequency weighting. Different domain (generation diversity) but same core mechanism.

### Continuous latent language models
- **Shao et al.** *Continuous Autoregressive Language Models (CALM).* arXiv:2510.27688 (2025). [[blog]](https://shaochenze.github.io/blog/2025/CALM/) ~75M-param stacked-MLP autoencoder compressing K=4 tokens into 128d. Hit >99.9% reconstruction on English text. Closest prior art; our AE is ~17M params doing harder structured-data reconstruction with similar fidelity.
- **CoDAR.** Shen et al. 2026. Identified the pointwise discretization bottleneck that killed our run 7 diffusion approach — motivated the shift to continuous latent pipeline.

### Latent space translation (for the translator, upcoming)
- **Maiorca et al.** *Latent Space Translation via Semantic Alignment.* NeurIPS 2023. [[paper]](https://papers.neurips.cc/paper_files/paper/2023/file/ad5fa03c906ca15905144ca3fbf2a768-Paper-Conference.pdf) Closed-form Procrustes-style alignment between pretrained encoders. Orthogonal/affine transformations often outperform learned non-linear maps. Planned first approach for our translator.
- **Moschella et al.** *Relative Representations Enable Zero-Shot Latent Space Communication.* ICLR 2023. Alternative: represent latents relative to anchor samples. Zero-shot, no training.
- **Lähner & Moeller.** *On the Direct Alignment of Latent Spaces.* 2024.

### Architecture components
- **RoPE** (Rotary Position Embedding). Su et al. *RoFormer.* 2021.
- **Pre-norm transformer** with residual connections. Xiong et al. 2020.
- **FactoredEmbedding** (our own, carried from run 7). Low-rank factorization of vocab × d_emb.

## Infrastructure / Tools

New scripts added for checkpoint surgery (enable iteration without retraining from scratch):

- `training/edit_layers.py` — Net2DeeperNet for transformer depth. Identity-initialized layer insertion at append or interleave positions. Contract by removing evenly-spaced layers.
- `training/edit_convs.py` — drop middle conv layer, adjust remaining strides to preserve total compression.
- `training/expand_latent.py` — widen `to_latent` / `from_latent` projections via Net2Net (vestigial now that we removed the projection entirely).

Training script (`train_ae.py`) has full resumability:
- Mid-training: checkpoint per batch with `sig_state` signal handling
- Mid-validation: `val_state` persisted per chunk, resume picks up from last completed sample
- Post-epoch: `epoch_complete=True` advances to next epoch on resume
- Single-shot noise schedule triggers on plateau, persisted in checkpoint

All tools operate on the same checkpoint format, auto-detect architecture from state_dict shapes and metadata (`conv_strides`, layer counts, channels).
