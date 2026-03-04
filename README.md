# multimodal-classification
Train and dev are very different
Train
pos_ratio = 0.3588
neg = 5450
pos = 3050
Dev
pos_ratio = 0.5000
neg = 250
pos = 250

We observed that the training set contains 35.9% positive samples, while the validation set is perfectly balanced (50% positive). This distribution shift leads the model to favor the negative class during training, resulting in limited validation accuracy (~55%). To address this, we introduced class-weighted loss to compensate for class imbalance.