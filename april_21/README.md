# Information about the used samples

## Length of each preprocessed file & Distribution of flavour (training / validation / test) & Raw distributions of inputs (just one file)

Can be found in `information_datasets.ipynb`, where the length and flavour distributions are based on all targets, and input variables are based on training / validation / test samples from one file only. Lengths are saved in `.npy` arrays (to be used for attacks for example), true flavour distributions are saved to `.png` (formerly `.svg`), raw input variables are just displayed in the notebook, as the full distributions will be done later when comparing raw and disturbed inputs.

## Default values

### Percentage of defaults per variable and per flavour

To check the distribution of default values (track / secondary vertex), all inputs will be read and checked for the given `minima - default` value (this will be split by flavour and later on combined to a total number). To do this, use
```shell
sbatch < ~/aisafety/april_21/files_defaults.sh
```
which calls `files_defaults.py` with the standard number of files (49). Adjust the default value in the `.sh`-script according to the way the defaults were handled during cleaning and preprocessing. (Actually, this shouldn't matter, because it's just the default value that differs, not the definition of *being* a default value. So the percentage of defaults should stay the same.)  
As result, there will be three `.png` files with jet, track and secondary vertex variables and their respective percentages of defaults, split by flavour and total.