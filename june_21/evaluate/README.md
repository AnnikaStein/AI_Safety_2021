## AI Safety 2021
### Code `june_21/evaluate`
---
#### Annika Stein, last updated: 29.10.2021


In this directory, you will find the code with which I created the plots for the thesis. First, I describe the core modules to do the evaluation (...`.py`), and later I show how to use the scripts yourself (either interactively via e.g. `.ipynb` or as jobs with `submit_XYZ.py` & ...`.sh`, but that should be much easier to understand after having practiced submitting training jobs already). It is recommended to follow the instructions in the order presented here; if you don't want to read the full explanation first, read at least up until the point that matches what you want to execute such that you can understand the behaviour. The explanations simply become shorter, although the code might get more complex, because it contains more of what has been discussed already (didn't want to repeat things and maybe you simply want to look at the code yourself and figure out what happens in detail, the more experienced you become). Always make sure to adjust all paths in all scripts - I will not mention it everytime but it is meant implicitly.

### How the code works

#### `definitions.py`
Collection of useful definitions for plotting, regarding DeepCSV variables (and their short names), indices for jet, track and secondary vertex variables, or integer variables, ranges for plotting histograms of the inputs later, the units of the quantities, how many digits will be used when quoting the granularity of the binning, and if the histograms count 'Jets' or if 'Tracks' suits better. You don't need to *call* this script anytime for anything outside of what will be described in the following.

#### `plot_loss.py`
Used to create a plot with training and validation loss (over epoch). Can be done with several trainings at once, all given by their weighting method and with an identifier that concerns the number of files and number of jets used during training (to identify the correct directory to read the checkpoints). Several trainings (with potentially different weighting methods) are split by an `+`, potentially different numbers of jets are specified with `_<number of jets one training>,_<number of jets other training>` and so on, if all jets were used, the number of jets is -1. There is an example in line 21. All information on the focusing parameter, potential parameter for the adversarial training etc. is part of the long name of the weighting method, no need to specify this in another argument (*Note: this differs from the training script where all of these parameters are read individually*).

Lines 24-40: used to split the long names and setup and store the different specified parameters inside some lists (e.g. if you use the pattern explained above, the script will automatically grab the necessary information that's part of the long weighting method here). In case some parameters are not present in the name of the weighting method, one can ignore what's written into the respective lists (most often, this will happen to alpha because we normally don't vary this), they will not be used further in that case, because the actual text (see below) and the reading of the checkpoints from disk use the given long name, not the text.

Lines 42-59: used to build the text that will go into the legend of the plot by creating formatted strings based on what was submitted as parameters of the weighting method. The long names are used inside dictionaries that contain entries for all currently relevant cases, and the `f'` formatting ensures that there are different entries for different parameters, even though the overall setup is similar (e.g. if you compare two different focusing parameters and have everything else identical, this will create two entries in the `more_text` list, but the pattern would match between the two, so they would be created in the same line). `wm_def_text` contains other cases, in particular the default ones (e.g. without specifying a particular $\gamma$, this assumes $2$).

Line 61: colours which have dark and a bit more light colours alternatingly, such that for one weighting method, the colours are roughly similar and only differ by their 'lightness' or 'gradation'.

Lines 63-75: you need to adjust this to how far you have trained the models. You can also add more lines in a similar pattern if you use different setups than the ones shown here.

Lines 77-93: the figure is created, there is a loop over all weighting methods (`wm`). Inside, the lists are filled with all loss values for training and validation of one `wm` at a time, by using the inner loop that walks through all checkpoints and reads the relevant keys. Make sure to adjust the path. The plots use the colours and labels according to `k` being even or odd (training / validation).

Lines 95-102: some basic styling, and the plot is saved in three formats. Comment out what you don't need or keep all three, but adjust paths in any case.

#### `eval_inputs.py`
Can plot distributions of input variables (summed / split by flavour, custom range / min/max, log / linear all with the same execution), calculates KL divergence between raw/distorted, and utilizes `definitions.py` as well as the attacks we have so far.

Lines 1-36 should be known, with exception of `coffea.hist` (see later). For Lines 25 and 29: update paths.

Lines 38-44: if you wonder why we set again seeds, that was done to have the "same" "random" results for all checks with Gaussian noise (makes debugging easier, you might want to comment this out some time). For now I believe that's random enough. 😉 This will also influence loading random paths to subsets of the test samples, see later.

Lines 46-106: might need some update later, as some arguments have become redundant in the current implementation. Don't worry, if you stick to the guidelines explained later, it will not be a problem to have more than the necessary arguments there. (E.g. only using one weighting method, parameter for the distortion and so on instead of having lists of more than one setup, and `fixRange` is not relevant as well anymore because this will be done always currently). You can leave that part as it is now. The idea with the dictionaries that contain the labels is similar to what has been explained before for plotting the loss. Make sure your weighting methods are not completely different and can be identified with the patterns that are already implemented there. Otherwise, you can add something new.

Lines 109-112: nothing fundamentally new, but change the paths, as always.

Lines 114-131: with this one can do the evaluation on the full test set or only on smaller subsets, based on the condition this will then only store a limited number of the paths that point to the samples.

Lines 134-158: similar to earlier explanations; new is the definition of a list that will hold the KL divergences (`relative_entropies`).

Lines 161-182: load correct model (based on weighting method etc.) with criterion (loss function). Update the path.

Lines 184-end: the actual plotting function. It looks much, but the structure is split into several, almost identical parts that all create a slightly different plot. The function needs the variable to plot, the type and magnitude of the attack, information about custom ranges for min/max (this will be supplied based on `definitions.py`, see lines 1145-1147) and gets the information whether to use the reduced FGSM attack (this is normally true).

The first half (until line 682) uses min / max as based on *all* values found inside the data, the second half of the function insted uses custom ranges (defined externally, as mentioned before). Both cases also contain two different basic types of plots, a first option has all flavours combined into one histogram, and another splits the histograms per flavour. These setups would use linear y axes first, and yet another plot is made in every case with a log-scale. If you know this pattern, now you basically only need to understand about one eighth of the code, the rest is just slightly adjusted (copy/paste). 🥳

Lines 186-234: basic setup for labels; read in test samples and perform attack (actually, this will run several times as a loop, with a specified number of magnitudes of the attack, where a magnitude of 0 means don't disturb the inputs). More on the application of the code and how to specify this in detail follows later. There is a check that makes sure already existing samples are not overwritten, instead the new ones are then concatenated with the previous ones (like, appending to a list).

Lines 238-264: minima and maxima are computed from all available samples (also the distorted ones), and for the histograms, the range is enlarged a bit. Depending on whether the quantity is an integer variable or not, the histograms either use bins that correspond to integer values or take 100 bins, equally distributed between the min and max, but taking into account that the bin edges and centers differ (centers are needed for ratio plot, but edges when defining the `hist.Bin` for integer variables). For now, I understand `hist.Cat` as 'what' you are filling into the histogram, and `hist.Bin` means 'how' in that case (although I think it has more to do with one thing being categorical and the other being quantized). But I'm definitly not an expert for explaining coffea, so please refer to the [documentation](https://coffeateam.github.io/coffea/index.html) or just ask Andrzej. 😉 Also, I did [this tutorial](https://github.com/CoffeaTeam/coffea-hats) ~1 year ago (has been updated since then), so maybe have a look if you want to learn more.

Lines 266-317: now the histogram(s) actually get filled (raw or disturbed, and at this stage, still split per flavour), and there is rather boring code to get the right units baked into the labels.

Lines 319-384: bit more interesting, because now, the relevant histograms are combined to get sums over all flavours, there is the collection of the numerator and denominator for the ratio plot (with stat. error) and the KL divergence is calculated. (To avoid division by zero: set the problematic cases to a small, non-zero number of entries in line 337). Then the relevant histograms or arrays of ratios are placed in the axes defined at the beginning (feat. some additional styling). To save the plots and the KL divs, update the paths. (Not going to repeat this from now on I think, it applies *everywhere*).

Lines 387-459: log plot of the previous one, nothing new

Lines 461-588: only difference compared to the first plot is that now, the ratio plot will not be used, but the main panel splits the histograms per flavour (for this, the `overlay='flavour'` is used). Ratios are calculated, but that would not be necessary as we are not plotting them. Could be commented out of course. 

Respective log plot follows.

Second half, as explained, uses not the automatically calculated minima and maxima, but the custom ones. So the definition of the bins changes marginally. Other than that, the same four plots.

#### `eval_discriminator_shapes.py`
Creates plots for four outputs and four discriminators as histograms that contain either the full sample or split by flavour. Starts similar to what has been explained so far (weighting methods, loading test samples...) but on top, also DeepCSV is loaded. Also, because we want to plot defaults not too far away from the other possible values (aspect ratio of the plots) the DeepCSV defaults are placed closer to zero (lines 206-209).

Lines 220-290: There are four functions that calculate the discriminators, making sure not to divide by zero.

Lines 298-end: several types of plots are created, but all without keeping gradients of tensors in memory. (Only done with raw quantities the entire time.) Can be used in three versions: to compare epochs and calculate some stats (KS test, KL div), compare weighting methods (not fully implemented currently) or to get detailed plots for one epoch, one weighting method only. No need to understand everything here in detail now, but you can of course compare the code to what we have discussed so far - it's comparable (e.g. using coffea histogram feature to have things organized in categories and bins). Maybe have a more closer look at the big `else` case instead, this is the one which I used a lot. But again, you should be able to understand what's going on there if you followed through either the previous explanations or even did the coffea-hats. There are two different histograms, one for the outputs, one for the discriminators. Outputs are plotted in three versions (custom tagger not stacked & DeepCSV stacked, both stacked, both stacked, but custom tagger still shows the different contributions that get summed), discriminators in two (not stacked or stacked custom shapes on top of stacked DeepCSV).

#### `eval_roc_new.py`
(Ignore the old version `eval_roc.py` by the way.)

This can be used for several purposes, mainly to plot ROC curves, but also to *just* get the area under the ROC curve for many epochs (i.e. without getting plots, only the evaluation of that quantity itself).

Again, the first ~220 lines are similar to other scripts explained already.

There are two main cases inside this script: doing no comparison with other weighting methods (only with DeepCSV, if one wants to) or comparing different custom models / trainings. That's the check in line 225. `force_compare` has been introduced to use the more advanced plotting as part of the second case also when comparing the custom model with DeepCSV, so it is actually preferred to use `force_compare` if one only has one individual custom model (and not several weighting methods, epochs, distortions...) because the plots will look better. More on how to call the script later. This also means that the more relevant code starts at line 665, so you can skip what's in between.

Actually, what takes a lot space there is the derivation of the style based on what has been requested for plotting, concerning colours, labels, linestyles, text / headers (lines 681-882). Uncomment lines 879-882 is you only want to know if your requested plotting setup is ok.

Lines 883-977: inputs for testing are loaded, together with targets. According to what was requested, the corresponding inputs are selected if one only does binary classification with discriminators. Already there, DeepCSV discriminators are calculated.

Lines 981-1123 have again some style specifications that concern the plot with its panels, for all relevant cases, the ranges for plotting have been manually checked.

Lines 1126-end: the loop does the evaluation (with the corresponding model) and plots the curves for every combination, as specified. Predictions are done with raw or distorted inputs. Slightly different styling (compared to the normal case for `compare`) is applied if there is only custom versus DeepCSV, customA versus customB... and so on. The ROC & AUC is first calculated and also plotted for one of the eight cases (b,bb,c,udsg,BvL,BvC,CvB,CvL), first only for the custom model that matches the current iteration of the loop, but also for DeepCSV, if that has been specified. Depending on whether one wants to save the plot or only keep the (potentially many) AUCs, the saving options differ a bit and again, the ROC curves are styled a bit further.

### How to execute the code yourself
In theory, with the right jobscripts, you could run all the core functionalities described so far inside a job. For some tasks, there are already such scripts, while for other things, I found the interactive method a bit easier (and sufficient). So now, this section is ordered by the 'problem' you want to want to solve, and below every item I list my current way of getting the results.

Create some directories that will hold the plots: `auc`,`confusion_matrices`,`discriminator_shapes`,`inputs`,`loss_plots`,`roc_curves` for example inside your `june_21/evaluate` directory. Compare with the paths in the scripts if you need to add more directories inside (one level deeper).

#### Loss plots
This can be done fully interactively, no need for a job (the script basically only reads checkpoints, no inputs...). Everything you need to know is part of `run_plot_loss.ipynb`, which has an example with the expected output inside if you want to compare e.g. basic and adversarial training. Adjust to your use case if necessary and make sure to execute the command or cell in the notebook with an activated conda environment.

#### Inputs
There is an interactive version (`run_eval_inputs.ipynb`) where the code is executed on a small subset of the inputs, but there is also the option to submit the full set of test inputs via `submit_eval_inputs.py` + `eval_inputs.sh`. Look at the interactive version first to make sure your code would run without errors (warnings are ok, could also be deactivated, they occur when there is division by zero for the ratios or when scaling the y axis, nothing to worry about). Each cell in the notebook has its own variable for plotting. Also comes with commands to create tar archives with all plots so that if you want to copy them over or download them at once, you just get them all grouped together by file type (svg or pdf).

The job(s) can be submitted for several variables at a time, currently the user will not be asked if the submission is ok (adjust lines 70-72 if you want to look at the submit command before actually submitting it). General advice: start submitting one variable first, with noise. That is the easiest setup that takes the shortest time and would tell you early that something is not right (if everything is ok, submit the whole set of inputs, with noise and FGSM). When you look at the arguments for `submit_eval_inputs.py` you will notice that even if you only want to submit a single variable, you need to pass a comma-separated list, like `0,0` if you want to submit the first variable only (Jet-$\eta$). When you submit more than one variable, both the first and second number are inclusive and every variable in between. Besides that, previous explanations regarding paths, specifying arguments for the submission script and the general structure of the submit-script apply here as well.

#### Discriminator shapes
Currently, only the interactive version is necessary to make the point. Could be extended to the full set, but you will notice that so far, there is no dedicated submit script to do this. So stick to `run_eval_disc.ipynb` and look at the commands inside, that's all you need for now. You'll also see that depending on what you want to plot exactly, you might have to adjust the code inside `eval_discriminator_shapes.py` depending on the focusing parameter (there are currently the cases 25 and 2 implemented, where the location of the legend differs to not have the text overlapping with the histograms, make sure you adjust the code for all occurrences of `# gamma25` etc. if you need to place the text somewhere else, but this is only for the styling, the code will run in any case). Check if the commands are using the same setup as what you used to train the models.

#### ROC curves
Basically the same as explained for the discriminator shapes applies here as well, relevant commands to get the plots are placed inside `run_eval_roc.ipynb`, starting with v2 (05.09.2021). The commands above are outdated / were used before a revision of the code, that mainly adjusts the styling of the lines. You only need to check if the weighting methods make sense in your case or if you used other parameters somewhere, then you need to adjust the commands a bit. Again, the job submission is not supported yet - you could extend the available skeleton to include all current options for plotting if you really need more than 13Mio. test samples - if not, you don't have to adjust anything and can just stick to what we have so far.

#### AUC
Here, the idea is to first get the AUC values and then create plots for them (two different steps, because by doing so, you store the values first and can plot them however you like, with interactive jupyter notebooks). Therefore you have to look at two notebooks: one to calculate the AUCs `run_auc.ipynb` (could later also be extended to a jobscript, see explanations above) and one to plot them as a function of epoch `plot_auc.ipynb`. The first notebook mentioned has the commands to get AUCs, but they are not yet updated to the newest version of the core module that does the evaluation. You will identify a couple of arguments are missing, they are not really relevant for the calculation of the AUC, but necessary to run the newest version of the code. Just add what's missing, I think you can find these yourself (hint: they are for styling, and you could look at the newest commits to find the differences - being able to inspect the history via git will be very useful anyway).

The plotting code itself (`plot_auc.ipynb`) should be ready for usage, except for changes you'd have to make to have your personal models there (paths, weighting methods...).