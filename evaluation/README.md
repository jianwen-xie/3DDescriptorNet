# Evaluation
We use a [volumetric CNN](https://github.com/charlesq34/3dcnn.torch) as the reference network for evaluation. 
The code is implemented in [Torch 7](http://torch.ch/docs/getting-started.html), please check required [torch packages](https://github.com/charlesq34/3dcnn.torch) for running the code.

## Inception Score

To run the inception score on synthesized results for all categories.
- Download the [pretrained model](https://shapenet.cs.stanford.edu/media/3dnin_fc.zip) for 3D Object Classification and save it to current directory.
- Resize the synthesized results to shape of 30x30x30 and convert it to binary value {0, 1} with threshold 0.5.
- Run `inception_score.lua` on sampled results.
```bash
th inception_score.lua -syn_path sample_results.mat
```

## Softmax class probability

To compute softmax probability on synthesized results for single category (night stand for example), run
```bash
th average_probability.lua -syn_path ../output/night_stand/synthesis/sample2990.mat -class night_stand
```
