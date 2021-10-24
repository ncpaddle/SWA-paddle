## Stochastic Weight Averaging (SWA) - Paddle Version  

This repoitory contains a **Paddle** implementation of the Stochastic Weight Averaging(SWA) training method.

* Original Paper: [Averaging Weights Leads to Wider Optima and Better Generalization](https://arxiv.org/abs/1803.05407)

by Pavel Izmailov, Dmitrii Podoprikhin, Timur Garipov, Dmitry Vetrov and Andrew Gordon Wilson.

* Original Pytorch Implementation: [https://github.com/timgaripov/swa](https://github.com/timgaripov/swa)

## Introduction  

We implement the SWA method with [Paddle](https://github.com/PaddlePaddle/Paddle) and test with VGG16 model. The results  are close to the orginal paper on the CIFAR-10 datasets. 

A brief introduction about important folders:

`diff`: log and fake data from recommended procedures of reproducing research papers [论文复现指南](https://github.com/PaddlePaddle/models/blob/develop/docs/ThesisReproduction_CV.md#4)

`reprod_log`: Important tools for checking the alignment accuracy of each step in the model reproduction process [reprod_log tool](https://github.com/WenmuZhou/reprod_log)

`swa-paddle`: paddle version of SWA

`swa-python`: pytorch version of SWA

## Results:  

| Method  |DataSet| Environment | Model| Epoch| Test Accuracy|   
| --- | --- |--- | --- |---|---|  
| SWA| CIFAR-10| GeForce RTX 3090 | VGG-16 | 225 | 93.60 |  

## Reprod_Log:  

`forward_diff`: [forward_diff.log](https://github.com/ncpaddle/SWA/blob/master/diff/forward_diff.log)  
`metric_diff` : [metric_diff.log](https://github.com/ncpaddle/SWA/blob/master/diff/metric_diff.log)  
`loss_diff` : [loss_diff.log](https://github.com/ncpaddle/SWA/blob/master/diff/loss_diff.log)  
`bp_align_diff` : [bp_align_diff.log](https://github.com/ncpaddle/SWA/blob/master/diff/bp_align_diff.log)  
`train_align_diff` : [train_align_diff.log](https://github.com/ncpaddle/SWA/blob/master/diff/train_align_diff_log.log)  
`train_log` : [train_log.txt](https://github.com/ncpaddle/SWA/blob/master/diff/train_log.txt)
## Model:  

The model we have trained is save to : [Baidu Aistudio SWA Paddle](https://aistudio.baidu.com/aistudio/datasetdetail/113323)  
