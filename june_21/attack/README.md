## AI Safety 2021
### Code `june_21/attack`
---
#### Annika Stein, last updated: 27.10.2021

<div align="center">
<img src="https://miro.medium.com/max/1400/1*yYmoweUD-Yx3Vi4tEBRAeA.png" width="400"/><br>
    Fig. 1 <a href="https://towardsdatascience.com/know-your-enemy-the-fascinating-implications-of-adversarial-examples-5936bccb24af">[from towardsdatascience.com]</a>
</div>

<br>

Currently, there is only one script that contains all the code to attack the inputs: `disturb_inputs.py`. It only needs numpy and pytorch and the functions defined inside will only be used when called from within other scripts, when evaluating or when doing adversarial training.

#### `apply_noise`
This function takes the sample to be disturbed (jets with 67 features, you have them stored as `.pt` and they will be loaded in the right format when calling this function). The magnitude of the attack is given by `magn` (that's the $\sigma$ of the Gaussian distribution), there could be some offset ($\mu$), but for now we keep this at zero. The default does not have to be modified, and the device is necessary to tell the attack where the noisy data need to be placed in order to not mess up with the device of the model.

Modify line 5 to point to your scalers.

For noise, we don't need gradients (line 6).

Until line 10, the loading of the default values shuold be known to you, but make sure to again change the path.

Get the device (line 12).

Lines 13-20: some old code when this was not yet externalized. Ignore.

Lines 22-23: create a new Tensor with noisy values for each jet and each feature, and put the tensor on the device. Then, add this noise array / Tensor on top of the original sample.

Lines 24-25: not necessary anymore, because in this stage, we don't want to scale back yet.

Lines 26-39: don't perturb
- integer variables
- defaults
    - with large values
    - smaller values
    - very small values
    
Note how the floating point error differs a bit for the different variables. A task might be (as written somewhere earlier for one of the cleaning / preprocessing steps) to instead have a boolean mask stored for every jet & feature such that this inconvenient code can be simplified (and on top of that: this would ensure we always get *exactly* the right bins with defaults, no matter how good the floating point numbers approximate them after scaling back and forth...)
    
Line 41 just returns the noisy samples.

#### `fgsm_attack`
The magnitude of the attack is given by `epsilon`. This function takes the sample to be disturbed (similar to noise), but it also needs the targets for the systematic attack. You also need to submit the model and the criterion when calling the function from outside, unless they exist globally for the script from which the attack is called. `reduced` can be True or False and is used to specify whether the attack shall be used for all features, or only for those that 'make sense' with slightly adjusted values (keep True).`default` and `dev` are similar to noise.

Lines 44-49 are similar to noise. Again, modify the path.

Lines 51-556: get dedive, get a copy of the samples and set their `requires_grad` parameter to True, because we need to calculate gradients of the loss funciton *with respect to the inputs*.

Lines 60-62: not the best style, but with that one can call the function also without giving the model and criterion, if they exist outside. This trick has been introduced to use the same attack for evaluation, as well as for the adversarial training, without having to code the function again.

Lines 65-72: evaluate the model on the (currently undisturbed) inputs. Reset gradients. Do backpropagation with the loss.

Lines 74-79: again, we don't need gradients of everything, except for the those w.r.t. inputs (remember, they are specified in line 56). Calculate the direction into which the inputs should move (that's a tensor that consists of -1 or 1 entries per feature per jet) by taking the gradient of the loss w.r.t. inputs, and only storing the sign of that gradient. Modify the inputs by adding this, scaled by epsilon, to the inputs.

Lines 82-95: in case the reduced variant is used, don't perturb
- integer variables
- defaults
    - with large values
    - smaller values
    - very small values
    
(same description as for noise applies here as well).
    
Line 96 just returns the adversarial samples.

#### Tasks / ideas
- Feel free to add more attacks,
- combine the two attacks into one,
- create an iterative version of the FGSM...

once you have reproduced what we have so far. 😉
