| Model | Steps | Gen. PPL ↓ | Entropy | Seconds / 256 samples |
|:--------------------------|:--------|:---------------|----------:|:------------------------|
| Real TinyStories text     | –       | 6.89           |      3.98 | –                       |
| NCM (11 levels predicted) | 11      | 154.93 ± 3.59  |      3.97 | 2.1                     |
| Masked diffusion, T=8     | 8       | 138.21 ± 10.48 |      4.1  | 2.1                     |
| Masked diffusion, T=16    | 16      | 90.98 ± 7.97   |      4.08 | 3.8                     |
| Masked diffusion, T=32    | 32      | 68.78 ± 4.77   |      4.07 | 7.4                     |
| Masked diffusion, T=64    | 64      | 61.07 ± 3.93   |      4.06 | 14.6                    |
| Masked diffusion, T=128   | 128     | 57.49 ± 3.35   |      4.07 | 29.1                    |
| Autoregressive            | 128     | 14.71 ± 0.52   |      4.02 | 6.5                     |

Official released NCM (~89M params, full training): Gen PPL 16.52, entropy 3.97 (paper: 15.41 / 3.98).
Backbone 6 layers x 384; training seeds [0, 1, 2]; sampling seeds [0, 1, 2]; RTX 4070 Laptop GPU.
