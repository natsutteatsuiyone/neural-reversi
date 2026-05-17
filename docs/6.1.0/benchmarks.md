# Benchmarks

## AMD Ryzen 9 9950X3D

### Environment

- **CPU:** AMD Ryzen 9 9950X3D
- **Threads:** 32
- **Hash size:** 2048 MB

### Evaluation Accuracy (depth 15)

[`benchmarks/hard-30-depth15.md`](benchmarks/hard-30-depth15.md) — 289 positions from `hard-30.obf` solved at depth 15.

- Total time : 3.117s
- Total nodes: 170,234,809
- NPS        : 54,611,449
- Top 3 move : 100.0% (289/289)
- Score ±3   : 87.9% (254/289)
- Score ±6   : 99.3% (287/289)
- Score ±9   : 100.0% (289/289)
- MAE        : 1.63

### Endgame Solving

Problem files are located in [`problem`](../../problem/).

| Test                     | Time (6.1.0) | Time (6.0.0) | Speedup |
|:-------------------------|-------------:|-------------:|--------:|
| FFO #40–59               |      6.764s  |      7.659s  |  +11.7% |
| FFO #40–59 (CPB Enabled) |      5.974s  |      6.773s  |  +11.8% |
| FFO #60–79               |    217.349s  |    235.436s  |   +7.7% |
| FFO #60–79 (CPB Enabled) |    193.064s  |    209.945s  |   +8.0% |
| Hard-20                  |      2.353s  |      2.580s  |   +8.8% |
| Hard-25                  |     28.225s  |     32.421s  |  +12.9% |
| Hard-30                  |    815.125s  |    904.934s  |   +9.9% |
| Small-35                 |  9,570.437s  | 10,494.203s  |   +8.8% |

#### Details

| Test                                                   | Problems | Depth |          Time |             Nodes |           NPS |
|:-------------------------------------------------------|:--------:|:-----:|--------------:|------------------:|--------------:|
| [FFO #40–59](benchmarks/fforum-40-59.md)               |       20 | 20–34 |       6.764s  |    12,898,842,946 | 1,906,994,139 |
| [FFO #40–59 (CPB Enabled)](benchmarks/fforum-40-59.md) |       20 | 20–34 |       5.974s  |    12,898,615,601 | 2,159,171,234 |
| [FFO #60–79](benchmarks/fforum-60-79.md)               |       20 | 24–36 |     217.349s  |   357,695,668,948 | 1,645,721,568 |
| [FFO #60–79 (CPB Enabled)](benchmarks/fforum-60-79.md) |       20 | 24–36 |     193.064s  |   357,800,331,652 | 1,853,277,097 |
| [Hard-20](benchmarks/hard-20.md)                       |      276 |    20 |       2.353s  |     1,887,280,214 |   802,103,526 |
| [Hard-25](benchmarks/hard-25.md)                       |      311 |    25 |      28.225s  |    47,378,924,636 | 1,678,609,980 |
| [Hard-30](benchmarks/hard-30.md)                       |      289 |    30 |     815.125s  | 1,508,107,259,823 | 1,850,153,908 |
| [Small-35](benchmarks/small-35.md)                     |       30 |    35 |   9,570.437s  | 15,189,558,541,184 | 1,587,133,157 |

## MacBook Air (M5, 2026)

### Environment

- **CPU:** Apple M5
- **Threads:** 10
- **Hash size:** 2048 MB

### Endgame Solving

| Test                                     | Problems | Depth |     Time |          Nodes |         NPS |
|:-----------------------------------------|:--------:|:-----:|---------:|---------------:|------------:|
| [FFO #40–59](benchmarks/fforum-40-59.md) |       20 | 20–34 | 26.127s  | 12,014,556,037 | 459,847,411 |
