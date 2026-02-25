# Benchmark Results

## Environment

- **CPU:** AMD Ryzen 9 9950X3D (no overclock)
- **Hash size:** 512MB

## Level 1 Match ([XOT](https://berg.earthlingz.de/xot/aboutxot.php))

### 3.0.0 vs 2.0.0

- Games: 21568 W: 10706 L: 10108 D: 754
- Ptnml(0-2): 2320, 361, 5119, 369, 2615
- Elo: +9.6 ± 4.5
- Disk diff: +7458 (+0.35/game)

## Endgame Tests

Problem files are located in [`problem`](../../problem/).

| Test       | Time (3.0.0) | Time (2.0.0) | Speedup |
|:-----------|-------------:|-------------:|--------:|
| FFO #40–59 |      9.8634s |      9.6388s |   -2.3% |
| FFO #60–79 |    261.4083s |    264.8354s |   +1.3% |

### Details

| Test                                      | Problems | Depth |        Time |             Nodes |           NPS |
|:------------------------------------------|:--------:|:-----:|------------:|------------------:|--------------:|
| [FFO #40–59](benchmark/fforum-40-59.md)   |       20 | 20–34 |     9.8634s |    15,584,659,870 | 1,580,049,463 |
| [FFO #60–79](benchmark/fforum-60-79.md)   |       20 | 24–36 |   261.4083s |   319,194,475,854 | 1,221,057,158 |
| [Hard-20](benchmark/hard-20.md)           |      276 |    20 |     4.1235s |     1,776,320,883 |   430,779,892 |
| [Hard-25](benchmark/hard-25.md)           |      311 |    25 |    46.2880s |    53,006,792,590 | 1,145,151,931 |
| [Hard-30](benchmark/hard-30.md)           |      289 |    30 | 1,066.7050s | 1,575,349,834,563 | 1,476,837,396 |
