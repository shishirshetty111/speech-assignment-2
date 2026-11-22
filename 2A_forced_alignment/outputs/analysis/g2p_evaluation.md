Test: 2000 english_us_arpa words not in the own model's training sample.

| G2P model | WER | PER | WER ignoring stress | PER ignoring stress | no output |
|---|---|---|---|---|---|
| own G2P (50,000 words) | 42.9% | 11.2% | 34.2% | 8.5% | 0 |
| pretrained english_us_arpa G2P | 8.7% | 2.1% | 6.7% | 1.5% | 0 |
