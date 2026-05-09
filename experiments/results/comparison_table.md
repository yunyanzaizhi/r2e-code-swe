# Sokoban History Management Experiment Results

| Strategy | Complete | Steps | Train SR last/max | Val SR last/max | Prompt mean | Step time | Late val mean±std | Late prompt | Late step time | Late valid ratio |
|---|---|---:|---|---|---:|---:|---|---:|---:|---:|
| K=3 | yes | 51/50 | 0.375 / 0.375 | 0.062 / 0.250 | 720.3 | 827.1 | 0.125 ± 0.044 | 720.3 | 825.0 | 0.964 |
| K=5 | yes | 51/50 | 0.125 / 0.375 | 0.062 / 0.219 | 841.2 | 975.3 | 0.062 ± 0.026 | 844.6 | 1038.4 | 0.962 |
| Full History | yes | 51/50 | 0.125 / 0.344 | 0.094 / 0.188 | 1096.8 | 1210.4 | 0.136 ± 0.039 | 1097.1 | 1226.7 | 0.978 |
| Structured Summary | yes | 51/50 | 0.188 / 0.375 | 0.094 / 0.188 | 542.7 | 767.8 | 0.104 ± 0.039 | 541.8 | 774.1 | 0.931 |
