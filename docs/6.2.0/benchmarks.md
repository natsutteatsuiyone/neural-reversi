# Benchmarks

## AMD Ryzen 9 9950X3D

### Environment

- **CPU:** AMD Ryzen 9 9950X3D
- **Threads:** 32
- **Hash size:** 2048 MB

### Evaluation Accuracy (depth 15)

[`benchmarks/hard-30-depth15.md`](benchmarks/hard-30-depth15.md) — 289 positions from `hard-30.obf` solved at depth 15.

- Total time : 2.911s
- Total nodes: 170,323,530
- NPS        : 58,520,368
- Top 3 move : 100.0% (289/289)
- Score ±3   : 87.9% (254/289)
- Score ±6   : 99.3% (287/289)
- Score ±9   : 100.0% (289/289)
- MAE        : 1.63

### Endgame Solving

Problem files are located in [`problem`](../../problem/).

| Test                     | Time (6.2.0) | Time (6.1.0) | Speedup |
|:-------------------------|-------------:|-------------:|--------:|
| FFO #40–59               |      6.619s  |      6.764s  |   +2.1% |
| FFO #40–59 (CPB Enabled) |      5.909s  |      5.974s  |   +1.1% |
| FFO #60–79               |    206.503s  |    217.349s  |   +5.0% |
| FFO #60–79 (CPB Enabled) |    184.503s  |    193.064s  |   +4.4% |
| Hard-20                  |      2.129s  |      2.353s  |   +9.5% |
| Hard-25                  |     27.325s  |     28.225s  |   +3.2% |
| Hard-30                  |    784.301s  |    815.125s  |   +3.8% |
| Small-35                 |  8,972.499s  |  9,570.437s  |   +6.2% |

#### Details

| Test                                                   | Problems | Depth |          Time |              Nodes |           NPS |
|:-------------------------------------------------------|:--------:|:-----:|--------------:|-------------------:|--------------:|
| [FFO #40–59](benchmarks/fforum-40-59.md)               |       20 | 20–34 |       6.619s  |     12,886,860,222 | 1,946,857,425 |
| [FFO #40–59 (CPB Enabled)](benchmarks/fforum-40-59.md) |       20 | 20–34 |       5.909s  |     12,975,594,240 | 2,195,733,797 |
| [FFO #60–79](benchmarks/fforum-60-79.md)               |       20 | 24–36 |     206.503s  |    339,862,636,700 | 1,645,802,805 |
| [FFO #60–79 (CPB Enabled)](benchmarks/fforum-60-79.md) |       20 | 24–36 |     184.503s  |    339,709,280,448 | 1,841,212,826 |
| [Hard-20](benchmarks/hard-20.md)                       |      276 |    20 |       2.129s  |      1,885,285,412 |   885,441,087 |
| [Hard-25](benchmarks/hard-25.md)                       |      311 |    25 |      27.325s  |     47,433,112,523 | 1,735,903,761 |
| [Hard-30](benchmarks/hard-30.md)                       |      289 |    30 |     784.301s  |  1,465,113,429,251 | 1,868,049,295 |
| [Small-35](benchmarks/small-35.md)                     |       30 |    35 |   8,972.499s  | 14,025,790,373,984 | 1,563,197,843 |

## MacBook Air (M5, 2026)

### Environment

- **CPU:** Apple M5
- **Threads:** 10
- **Hash size:** 2048 MB

### Endgame Solving

| Test       | Time (6.2.0) | Time (6.1.0) | Speedup |
|:-----------|-------------:|-------------:|--------:|
| FFO #40–59 |     21.244s  |     26.127s  |  +18.7% |

#### Details

| Test                                     | Problems | Depth |     Time |          Nodes |         NPS |
|:-----------------------------------------|:--------:|:-----:|---------:|---------------:|------------:|
| [FFO #40–59](benchmarks/fforum-40-59.md) |       20 | 20–34 | 21.244s  | 11,922,967,614 | 561,227,044 |
