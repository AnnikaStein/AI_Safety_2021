## AI Safety 2021
### Code `june_21`
---
#### Annika Stein, last updated: 27.10.2021

In this directory, the code is organized into different tasks. More details are given in the respective subdirectories. It should be noted that before going through the scripts in this directory, one needs to follow the steps explained in another [README for the `may_21` version of the code](../may_21/README.md).

For example, one of the earliest steps for all following investigations concerns the `preparations` of inputs and some related studies. In this subdirectory, you will find a notebook called `preprocessing_tutorial.ipynb` which covers the preprocessing step.<br>
Once all files are preprocessed, the training can be started. This is done with the help of the scripts in `train_models`. The corresponding tutorial / write-up is the notebook `training_tutorial.ipynb`.<br> 
When the training is finished (or whenever one wants to start looking at performance, distorted inputs and so on), the code inside `evaluate` can help.<br>
`attack` contains the currently available distortions that can be applied to the inputs. Though the code can in principle be used standalone, the attacks are currently only called from within other parts of the code (e.g. during evaluation or when training with adversarial inputs). The directory contains another readme with explanations. Refer to section 1.6 (+ chapter 4, later) of the thesis if you need more info.
