## AI Safety 2021
### Code `may_21`
---
#### Annika Stein, last updated: 25.10.2021

A lot of the code in this directory is outdated / deprecated (inside the subdirectories), however, with two exceptions: for the cleaning (`preparations/clean.py`) and reweighting (`reweighting_prototyping.ipynb`), this is still the newest version available and the resulting arrays are used also for the `june_21` version.

For the cleaning, there is a dedicated notebook (`preparations/cleaning_tutorial.ipynb`) which explains the code inside `clean.py` in a bit more interactive way.

For the reweighting, the weights can be downloaded as `.npy` arrays, one does not have to run the jupyter notebook (`reweighting_prototyping.ipynb`) manually again, although it is of course recommended to at least take a look at the calculation. When this prompt occurs while trying to view the notebook from the web
> Error loading file viewer.

one can just click "load it anyway", it's slightly more than 10MB (due to many graphics inside...)  
With the weights placed in a personal directory at the HPC, one can proceed to the following steps that are part of this repository's [`june_21` folder](../june_21/README.md).
