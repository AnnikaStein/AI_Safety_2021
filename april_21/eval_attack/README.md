# Evaluation
#### Performance of the model, information about the samples and attacks

## Raw performance

### 4. Loss over epoch
Create a plot of training and validation loss over epoch, if the training was already done with both weighting methods (loss or no weighting) up to the given number of epochs (e.g. here: 90). The first argument and last argument will only be necessesary to load the correct models (that were trained on e.g. 49 files where the default value is 0.001 below the average minima).
```shell
python3 ~/aisafety/april_21/eval_attack/plot_loss.py 49 90 0.001
```
### 5. Discriminator shapes (split per flavour or everything) and generation of confusion matrices
```shell
python3 ~/aisafety/april_21/eval_attack/eval_discriminator_shapes.py 49 94 '_both' '0.001'
```
Will create four histograms for each output (P(b), P(bb), P(c), P(udsg)) and save the whole figure as a `.png`. The outputs of the classifier (custom tagger) as well as the DeepCSV outputs are displayed.`_both` means that there will be
- a separate file that has the "No weighting"-method
- another file with the "Loss weighting"-method
- and a file that has both methods combined.

Even with all 49 files, everything can be done interactively, so it is not necessary to call
```shell
sbatch < ~/aisafety/april_21/eval_attack/eval_discriminator_shapes.sh
```
(but it is possible, of course). Then, one would have to modify the number of files (here: 49), the number of trained epochs (here: 94, which were done for both weighting methods), the mode for plotting (`'_as_is'` for no weighting alone, split per flavour; `'_new'` for loss weighting alone, split per flavour; `'_both'` does three things: plot for both methods together with all flavours summed and it also does the two methods separately.) The default value (average minima minus this value) is given as the last parameter, here '0.001'.

Limits of the axes are scaled automatically, taking every sample / overlay into account (one never has to modify the ylims again manually, yay!)

The script also generates the confusion matrices that compare predicted and true labels and stores them as `.npy`-files, which can be read later. (#ToDo)
### 6. ROC-curves (single epoch & DeepCSV or per epoch)


### 7. AUC (per epoch)

## Attacks

### 8. Input shapes (with Noise, FGSM, different parameters)

### 9. ROC (fixed epoch, different parameters or fixed parameter, different epochs)

### 10. AUC (fixed parameter, different epochs)

### 11. Correlations

### 12. KL-divergences

### 13. AUC-ranking
