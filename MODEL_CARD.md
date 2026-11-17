# Model card: pet breed classifier (ResNet-50, Oxford-IIIT Pet)

## Model
* **Architecture:** torchvision ResNet-50, ImageNet-1k V2 weights, new head
  (dropout 0.2 + linear, 37 outputs). 23.6M parameters. Input 224x224 RGB.
* **Training:** two-stage fine-tuning (head only for 1 epoch, then all layers with backbone LR =
  0.1x head LR), AdamW, label smoothing 0.1, warm-up + cosine schedule, TrivialAugment,
  early stopping. Best checkpoint at epoch 7 of 11 (validation accuracy 96.2%).
* **Config and history:** `reports/resnet50/config.json`, `history.json`.

## Intended use
Educational and portfolio demonstration of fine-grained classification, explainability and
MLOps practice. Not intended for veterinary, legal, breed-registry, insurance or any
decision that affects animals or people.

## Data
Oxford-IIIT Pet: 37 breeds (12 cat, 25 dog), about 200 images per breed. `trainval` split
(3,680 images) is divided 90/10 stratified into train and validation; the official `test` split
(3,669 images) is used once for the reported results.

## Performance (test split, 3,669 images)
Top-1 **93.0%**, top-5 99.5%, macro-F1 0.929, ECE 0.094. Cat vs dog accuracy 99.8%.
Per-breed accuracy ranges from 59% (American Pit Bull Terrier) to near 100% for distinctive
breeds; see `reports/resnet50/metrics.json`. Baseline from-scratch CNN: 60.0% top-1.

## Limitations and risks
* Only the 37 supported breeds. Any other animal, or a mixed breed, is still forced into one of
  them, often with high confidence. The demo warns when confidence is below 50%, but confidence
  is imperfectly calibrated.
* Weak on look-alike breeds (Pit Bull / Staffordshire / American Bulldog; Ragdoll / Birman).
  Breed labels for such dogs are inherently ambiguous; do not treat predictions as ground truth.
* Trained on a small, curated dataset of mostly well-framed photos. Expect lower accuracy on
  unusual poses, heavy occlusion, low light, puppies or kittens, and costumes.
* Breed identification from appearance of dogs is contested (breed labels are unreliable
  predictors of behaviour). Never use the output to judge an individual animal.
* Single training run; results carry seed variance that was not measured.

## Explainability
Grad-CAM on `layer4` is exposed through `/explain` and the demo. Heatmaps show where the model
looked, which helps spot background shortcuts, but they are a visualisation, not proof of
causal reasoning.

## Maintenance
Retraining is a single command (`make train-resnet`). Model files are not stored in this
repository; reproduce with the shipped configs.
