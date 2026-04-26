# Benchmarks

## AMD Ryzen 9 9950X3D

### Environment

- **CPU:** AMD Ryzen 9 9950X3D
- **Threads:** 32
- **Hash size:** 2048 MB

### Evaluation Accuracy (depth 15)

[`benchmarks/hard-30-depth15.md`](benchmarks/hard-30-depth15.md) — 289 positions from `hard-30.obf` solved at depth 15.

- Total time : 3.758s
- Total nodes: 151,563,290
- NPS        : 40,335,131
- Top 3 move : 100.0% (289/289)
- Score ±3   : 87.9% (254/289)
- Score ±6   : 99.3% (287/289)
- Score ±9   : 100.0% (289/289)
- MAE        : 1.64

### Endgame Solving

Problem files are located in [`problem`](../../problem/).

| Test       | Time (6.0.0) | Time (5.0.0) | Speedup |
|:-----------|-------------:|-------------:|--------:|
| FFO #40–59 |      7.659s  |      8.630s  |  +11.3% |
| FFO #60–79 |    235.436s  |    243.371s  |   +3.3% |
| Hard-20    |      2.580s  |      2.999s  |  +14.0% |
| Hard-25    |     32.421s  |     36.238s  |  +10.5% |
| Hard-30    |    904.934s  |    967.178s  |   +6.4% |

#### Details

| Test                                                  | Problems | Depth |          Time |             Nodes |           NPS |
|:------------------------------------------------------|:--------:|:-----:|--------------:|------------------:|--------------:|
| [FFO #40–59](benchmarks/fforum-40-59.md)               |       20 | 20–34 |       7.659s  |    12,785,599,475 | 1,669,384,514 |
| [FFO #40–59 (PBO Enabled)](benchmarks/fforum-40-59.md) |       20 | 20–34 |       6.773s  |    12,780,105,799 | 1,886,829,160 |
| [FFO #60–79](benchmarks/fforum-60-79.md)               |       20 | 24–36 |     235.436s  |   339,714,844,461 | 1,442,919,528 |
| [FFO #60–79 (PBO Enabled)](benchmarks/fforum-60-79.md) |       20 | 24–36 |     209.945s  |   342,035,698,205 | 1,629,165,443 |
| [Hard-20](benchmarks/hard-20.md)                       |      276 |    20 |       2.580s  |     1,681,608,922 |   651,747,905 |
| [Hard-25](benchmarks/hard-25.md)                       |      311 |    25 |      32.421s  |    46,126,798,879 | 1,422,746,901 |
| [Hard-30](benchmarks/hard-30.md)                       |      289 |    30 |     904.934s  | 1,471,667,735,637 | 1,626,269,909 |
| [Small-35](benchmarks/small-35.md)                     |       20 |    35 |  10,494.203s  | 14,526,843,918,418 | 1,384,273,243 |

## MacBook Air (M5, 2026)

### Environment

- **CPU:** Apple M5
- **Threads:** 10
- **Hash size:** 2048 MB

### Endgame Solving

| Test                                     | Problems | Depth |     Time |          Nodes |         NPS |
|:-----------------------------------------|:--------:|:-----:|---------:|---------------:|------------:|
| [FFO #40–59](benchmarks/fforum-40-59.md) |       20 | 20–34 | 27.399s  | 11,880,415,269 | 433,603,731 |
