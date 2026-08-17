# Benchmarks

## AMD Ryzen 9 9950X3D

### Environment

- **CPU:** AMD Ryzen 9 9950X3D
- **Threads:** 32
- **Hash size:** 2048 MB

### Evaluation Accuracy (depth 15)

[`benchmarks/hard-30-depth15.md`](benchmarks/hard-30-depth15.md) — 289 positions from `hard-30.obf` solved at depth 15.

- Total time : 2.448s
- Total nodes: 130,053,393
- NPS        : 53,124,216
- Top 3 move : 100.0% (289/289)
- Score ±3   : 85.5% (247/289)
- Score ±6   : 99.3% (287/289)
- Score ±9   : 100.0% (289/289)
- MAE        : 1.67

### Endgame Solving

Problem files are located in [`problem`](../../problem/).

| Test                     | Time (6.3.0) | Time (6.2.0) | Speedup |
|:-------------------------|-------------:|-------------:|--------:|
| FFO #40–59               |      6.029s  |      6.619s  |   +8.9% |
| FFO #40–59 (CPB Enabled) |      5.370s  |      5.909s  |   +9.1% |
| FFO #60–79               |    184.418s  |    206.503s  |  +10.7% |
| FFO #60–79 (CPB Enabled) |    164.525s  |    184.503s  |  +10.8% |
| Hard-20                  |      1.795s  |      2.129s  |  +15.7% |
| Hard-25                  |     24.473s  |     27.325s  |  +10.4% |
| Hard-30                  |    746.719s  |    784.301s  |   +4.8% |
| Small-35                 |  8,314.563s  |  8,972.499s  |   +7.3% |

#### Details

| Test                                                   | Problems | Depth |          Time |              Nodes |           NPS |
|:-------------------------------------------------------|:--------:|:-----:|--------------:|-------------------:|--------------:|
| [FFO #40–59](benchmarks/fforum-40-59.md)               |       20 | 20–34 |       6.029s  |     12,791,069,853 | 2,121,636,861 |
| [FFO #40–59 (CPB Enabled)](benchmarks/fforum-40-59.md) |       20 | 20–34 |       5.370s  |     12,736,659,249 | 2,371,617,478 |
| [FFO #60–79](benchmarks/fforum-60-79.md)               |       20 | 24–36 |     184.418s  |    334,416,880,379 | 1,813,367,928 |
| [FFO #60–79 (CPB Enabled)](benchmarks/fforum-60-79.md) |       20 | 24–36 |     164.525s  |    332,545,645,155 | 2,021,247,959 |
| [Hard-20](benchmarks/hard-20.md)                       |      276 |    20 |       1.795s  |      1,493,278,814 |   831,810,015 |
| [Hard-25](benchmarks/hard-25.md)                       |      311 |    25 |      24.473s  |     46,422,669,177 | 1,896,910,314 |
| [Hard-30](benchmarks/hard-30.md)                       |      289 |    30 |     746.719s  |  1,501,762,723,900 | 2,011,149,287 |
| [Small-35](benchmarks/small-35.md)                     |       30 |    35 |   8,314.563s  | 14,770,436,809,931 | 1,776,453,780 |

## MacBook Air (M5, 2026)

### Environment

- **CPU:** Apple M5
- **Threads:** 10
- **Hash size:** 2048 MB

### Endgame Solving

| Test       | Time (6.3.0) | Time (6.2.0) | Speedup |
|:-----------|-------------:|-------------:|--------:|
| FFO #40–59 |     20.071s  |     21.244s  |   +5.5% |

#### Details

| Test                                     | Problems | Depth |     Time |          Nodes |         NPS |
|:-----------------------------------------|:--------:|:-----:|---------:|---------------:|------------:|
| [FFO #40–59](benchmarks/fforum-40-59.md) |       20 | 20–34 | 20.071s  | 11,861,819,001 | 590,999,820 |
