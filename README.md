##  Multiple Semantic Prompt for Rehearsal-Free Continual Learning
PyTorch code for the paper:\
**Multiple Semantic Prompt for Rehearsal-Free Continual Learning**\
**_[Junwei Chen]_**, *Zhenyu Zhang, Depeng Li, Zhigang Zhang*


## Abstract
Deep neural networks have achieved impressive achievements in many applications while suffering from *catastrophic forgetting*, which refers to drastic degradation in performance on former tasks when training on novel ones. Continual learning aims to address the issue. A simple but effective solution is to replay a subset of previous data, while it increases memory costs and may violate data privacy. Recently, prompt-based methods have seen a recent explosion of interest because they can achieve good performances without the rehearsal buffer. 
In this work, we link prompts and textual labels to learn semantic information and combines prompts according to the similarity between visual and textual features. So we call it the Multiple Semantic Prompt (MSP). The combined prompt captures semantic information from learned tasks and can instruct the pre-trained model to perform better in zero-shot transfer. Extensive experiments show the superiority of our method over SOTA prompt-based methods by over 3\% in average final accuracy. Our method also has higher zero-shot classification accuracy than the vanilla vision language model by over 15\% on the established benchmark.


## Setup
 * Install anaconda: https://www.anaconda.com/distribution/
 * set up conda environment w/ python 3.8, ex: `conda create --name msp python=3.8`
 * `conda activate msp`
 * `sh install_requirements.sh`
 
## Datasets
 * Place data into the folder `data/`
 * **CIFAR 100**: should automatically be downloaded
 * **ImageNet-R**: 
    * the data can retrieve from: https://github.com/hendrycks/imagenet-r
    * Ensure the file `data/clip_label/imagenet-r_label.yaml` exists
    * ImageNet-R data should be arrange like below:
        ```
        data
        |
        └───clip_label 
        |   │   
        |   └───imagenet-r_label.yaml   
        │
        └───imagenet-r
            │   n01443537
            │   n01484850
            │   ...
            └───n12267677
                │   deviantart_0.jpg
                │   deviantart_1.jpg
                │   ...

        ```

## Training
All commands should be run under the project root directory. 

1-task, 5-tasks and 10-tasks setting for cifar100

```bash
sh experiments/cifar100-s1.sh
sh experiments/cifar100-s5.sh
sh experiments/cifar100-s10.sh
```

1-task, 5-tasks and 10-tasks setting for ImageNet-R

```bash
sh experiments/imagenet-r-s1.sh
sh experiments/imagenet-r-s5.sh
sh experiments/imagenet-r-s10.sh
```

## Results
Results will be saved in a folder named `outputs/`. To get the final average accuracy, retrieve the final number in the file `outputs/**/results-acc/global.yaml`
