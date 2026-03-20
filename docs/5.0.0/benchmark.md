# Benchmark Results

## Environment

- **CPU:** AMD Ryzen 9 9950X3D (no overclock)
- **Hash size:** 1024MB

## Level 1 Match ([XOT](https://berg.earthlingz.de/xot/aboutxot.php))

### 5.0.0 vs 4.0.0

- Games: 21568 W: 11021 L: 9791 D: 756
- Ptnml(0-2): 2242, 343, 4980, 381, 2838
- Elo: +19.8 ± 4.6
- Disc diff: +32686 (+1.52/game)

## Endgame Tests

Problem files are located in [`problem`](../../problem/).

| Test       | Time (5.0.0) | Time (4.0.0) | Speedup |
|:-----------|-------------:|-------------:|--------:|
| FFO #40–59 |      8.6303s |      8.8779s |   +2.8% |
| FFO #60–79 |    243.3709s |    241.2194s |   -0.9% |
| Hard-20    |      2.9986s |      3.5341s |  +15.2% |
| Hard-25    |     36.2379s |     42.2908s |  +14.3% |
| Hard-30    |    967.1777s |    981.4008s |   +1.4% |

### Details

| Test                                      | Problems | Depth |        Time |             Nodes |           NPS |
|:------------------------------------------|:--------:|:-----:|------------:|------------------:|--------------:|
| [FFO #40–59](benchmark/fforum-40-59.md)   |       20 | 20–34 |     8.6303s |    14,173,241,868 | 1,642,265,259 |
| [FFO #60–79](benchmark/fforum-60-79.md)   |       20 | 24–36 |   243.3709s |   310,896,777,518 | 1,277,460,771 |
| [Hard-20](benchmark/hard-20.md)           |      276 |    20 |     2.9986s |     1,831,696,902 |   610,850,698 |
| [Hard-25](benchmark/hard-25.md)           |      311 |    25 |    36.2379s |    50,164,842,818 | 1,384,319,809 |
| [Hard-30](benchmark/hard-30.md)           |      289 |    30 |   967.1777s | 1,469,966,788,583 | 1,519,851,821 |
