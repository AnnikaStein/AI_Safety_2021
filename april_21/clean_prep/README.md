## Cleaning and preprocessing
#### Using Nik's trick to shift defaults close to the minima of each distribution

### 1. Cleaning
#### Example:
```shell
conda activate my-env
cd /home/um106329/aisafety/april_21/clean_prep
voms-proxy-init --voms cms --vomses ~/grid-security/vomses
python3 clean.py 0 499 '0.001'
```
#### Explanation:
- 0 : the first `.root`-file to be cleaned
- 499 : the last `.root`-file to be cleaned
- '0.001' : by how much the new default value is smaller than the minima for each variable  

Do the above for every set of files separately (i.e. also do 500-999, 1000-1499, 1500-1999, 2000-2446), probably all in different screen sessions or on different nodes. Just make sure the proxy is always valid.

#### List of the steps that will be done:  
- Move all NaN, -Infinity, +Infinity, original -999 to the new defaults (the last item was added because there can be variables with a -999 default that was there without us doing anything, so they should be treated just like all other defaults and moved to the new deafult value.)  
- Cut all jets with DeepCSV outputs < 0 or > 1 (no valid probability)  
- Cut all jets where neither Jet N Selected Tracks, nor Jet N Secondary Vertices is > 0  
- For the n-th track variable (technically, the 0-th track variable is for the first track), check if Jet N Selected Tracks is high enough to actually reconstruct this variable, if not, e.g. if track-variable 0 is considered ('first track'), but Jet N Selected Tracks is = 0, this will be defaulted. Same for track variables 1 to 5 (so, for the second up to the sixth track).  
- For th n-th eta_rel variable, the trick is similar, but one has to check for the variable Jet N Eta_Rel instead and it's only 0 to 3, not 0 to 5.  
- For all variables that have AboveCharm in their name, move all -1 defaults to the new default value.

### 2. Preprocessing
#### Example:
```shell
cd /home/um106329/aisafety/output_slurm
sbatch < ~/aisafety/april_21/clean_prep/prep.sh
```
#### List of the steps that will be done:   
As specified in `prep.sh`, the cleaned inputs with the given new default value will now be preprocessed further and scaled. They are split into training-, validation- and testset. Additionally, for the test inputs we also keep the DeepCSV tagger outputs. Currently, the default is placed at 0.001, so in the function called `preprocess`, we use the `else`-case. Scalers are derived based on the `train_inputs`, but only those that were not defaulted. Then, train/val/test inputs are all scaled accordingly, including the defaults.