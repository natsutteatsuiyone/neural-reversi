# Benchmark Results

## Environment

- **CPU:** AMD Ryzen 9 9950X3D (no overclock)
- **Hash size:** 512MB

## Level 1 Match ([XOT](https://berg.earthlingz.de/xot/aboutxot.php))

### 4.0.0 vs 3.0.0

- Games: 21568 W: 10599 L: 10249 D: 720
- Ptnml(0-2): 2410, 367, 5075, 327, 2605
- Elo: +5.6 ± 4.5
- Disc diff: +7140 (+0.33/game)

## Endgame Tests

Problem files are located in [`problem`](../../problem/).

| Test       | Time (4.0.0) | Time (3.0.0) | Speedup |
|:-----------|-------------:|-------------:|--------:|
| FFO #40–59 |      8.8779s |      9.8634s |  +10.0% |
| FFO #60–79 |    241.2194s |    261.4083s |   +7.7% |
| Hard-20    |      3.5341s |      4.1235s |  +14.3% |
| Hard-25    |     42.2908s |     46.2880s |   +8.6% |
| Hard-30    |    981.4008s |  1,066.7050s |   +8.0% |

### Details

| Test                                      | Problems | Depth |        Time |             Nodes |           NPS |
|:------------------------------------------|:--------:|:-----:|------------:|------------------:|--------------:|
| [FFO #40–59](benchmark/fforum-40-59.md)   |       20 | 20–34 |     8.8779s |    14,613,676,238 | 1,646,073,535 |
| [FFO #60–79](benchmark/fforum-60-79.md)   |       20 | 24–36 |   241.2194s |   333,725,730,295 | 1,383,494,571 |
| [Hard-20](benchmark/hard-20.md)           |      276 |    20 |     3.5341s |     1,900,573,025 |   537,781,338 |
| [Hard-25](benchmark/hard-25.md)           |      311 |    25 |    42.2908s |    54,828,045,707 | 1,296,453,264 |
| [Hard-30](benchmark/hard-30.md)           |      289 |    30 |   981.4008s | 1,561,213,092,065 | 1,590,800,713 |
