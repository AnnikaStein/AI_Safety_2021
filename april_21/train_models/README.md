## Training
#### Using the newly cleaned and preprocessed TT samples

### 3. Training
#### Example:

```shell
cd /home/um106329/aisafety/output_slurm  # This is technically not necessary IF the submit-script below is used, if you decide to submit the job without the following script, make sure you know from where you called sbatch (this defines where the logs are placed).
python3 ~/aisafety/april_21/train_models/submit_training.py -f 10 -p 0 -a 30 -w '_new' -d '0.001'
```
With this script, it is possible to submit batch jobs for the training of the model with different setups (without the need to wait for a job actually starting before one can submit another job with another setup).
#### Explanation:
- `-f 10` : the number of files that will be used in the training (here, 10 TT files, the default would be 49)
- `-p 0` : the number of previous epochs that were already done in another training
- `-a 30` : the number of additional epochs that the model will be trained
- `-w '_new'` : the weighting method for the training, this is the recommended one that uses loss weighting
- `-d '0.001'` : this is how mcuh the default values differ from the average minimum of each variable (positive value, but is subtracted)  

If you just want the default setup
```shell
python3 ~/aisafety/april_21/train_models/submit_training.py  # this is equivalent to the line below:
python3 ~/aisafety/april_21/train_models/submit_training.py -f 49 -p 0 -a 30 -w '_new' -d '0.001'
```

The script `submit_training.py` handles the job-name (according to the parameters), calculates a time-limit for the job by multiplying a standard value that was based on all 49 files and 30 epochs by a factor relative to these standards and also calculates how much memory the job will most likely need (again based on multiplication of the standard value with a factor relative to the standard setup).  
For example, this is how the submitted batch job could end up like:
```shell
sbatch --time=15:00:00 --mem-per-cpu=41G --job-name=TTtr_10_0_30_new_0.001 --export=FILES=10,PREVEP=0,ADDEP=30,WM=_new,DEFAULT=0.001 /home/um106329/aisafety/april_21/train_models/training.sh
```

Of course, one could have also just written something like that manually (but then, make sure that you remember from where you submitted the job, as the output(log-)file will be placed in the directory from which the job was called).  
This procedure is done to ensure the time will not run out before completion of the job and that there is enough memory to laod the data (and the model, which will come with a computation graph...).  
On the other hand, one does not need to ask for more time and memory than what is needed, of course because of the accounting, but also because the job would take unneccesarily long to start (and block ressources - bad).
#### What will happen?  
First of all, the jobscript is completed by additional setups in `training.sh`, which will stay the same the whole time. In this script, the actual training (`training.py`) will be started. The training will use the specified parameters above, all using the newly cleaned and preprocessed TT samples.  

#### Key features of the training-script
The newest version will parse the parameters that were given during the job submission. If this is the first time that the training is done with this setup (i.e. `-p 0`), new directories will be created (their names reflect the weighting method, the number of datasets and which default value was subtracted from the minima). One directory will be in `/home` (slow but backups are done), the other in `/hpcwork` (faster and mounted when batch jobs are running, but no backups). They will contain the checkpoints of the model after every epoch (plus the model alone without additional `state_dicts` once the training-script finishes). Also, when it's the first time with this setup, the seeds will be set (that way in principle all setups should start with the same initial weights and sources of randomness will be reduced, which could be useful for debugging and reproducing stuff).  
The training can be resumed after any epoch by using the correct number of previous epochs, internally, all `state_dicts` will be loaded and used as a checkpoint (for the model and the optimizer).
To keep track of the progress, especially right after the training started, an additional logfile will be written (which is not the `SLURM` output).  
For the entire training (if not specified differently) one can keep the batch size constant, both for the training and the validation inputs. The learning that is given is just the initial learning rate (which will decrease the longer we train the model). It's not really adaptive currently, it's rather "decaying" in a fixed way (so, having a large number of stale epochs would not do anything to the learning rate currently, which could be improved maybe).  
Loading the datasets themselves is done by using `pytorch`'s functions `TensorDataset` (merges inputs and targets together) and `ConcatDataset` (merges several `TensorDataset`s).  
We use the `DataLoader` (again part of `torch.utils.data`) to create batches. If the weighting method `'_wrs'` is used, a `WeightedRandomSampler` is used to draw an equal amount of samples from the whole set of inputs, for this, class weights are used. These class weights are calculated internally with `n_total / (n_classes * n_in_this_class)`. Another way to make use of these weights is by choosing `'_new'` as weighting method, there we weight only inside the loss function. Using `'_as_is'` simply doesn't care about the imbalanced classes and applies no weighting at all, just a plain loss function.