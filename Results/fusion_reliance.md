# Reliance of the fusion models on each modality

Mean over 5 seeds. Swapped: each test file is given the text (or speech) of another test file.

| Model | Test setting | Normal | Text swapped | Speech swapped | Gate (share from speech) |
|---|---|---|---|---|---|
| fusion_concat | both speakers | 99.7% | 99.7% | 15.3% | - |
| fusion_concat | cross-speaker OAF->YAF | 52.7% | 52.6% | 14.5% | - |
| fusion_concat | cross-speaker YAF->OAF | 38.0% | 38.1% | 14.0% | - |
| fusion_gated | both speakers | 99.7% | 99.7% | 15.2% | 0.85 |
| fusion_gated | cross-speaker OAF->YAF | 56.6% | 56.6% | 14.3% | 0.82 |
| fusion_gated | cross-speaker YAF->OAF | 47.8% | 47.8% | 13.4% | 0.86 |
