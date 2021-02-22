| Order | Method | Device | Batch Size | Best Epoch | Max Epoch | Start LR | LR Decay | Optimizer | mIoU     | F1 Score   |
|:-----:|:------:|:------:|:----------:|:----------:|:----------|:--------:|:---------|:----------|:---------|:-----------|
|1      |CT_Unetwith1encoder|3090      |8           |           | 0.01     |          | Adam      |0.9815    |            |
|2      |unet                   |3090  |8           |    122    | 0.01     |          | Adam      |0.9824    |   0.9911   |
|2      |CT_Unetwith4encoder|3090      |8           |       | 0.01     |          | Adam      |    |      |