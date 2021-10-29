# Useful commands (for copy, paste)
## Create ROC curves example
```bash
python eval_roc_new.py 278 200 BvL_eps0.01,BvL_raw no _ptetaflavloss_focalloss_gamma25.0_adv_tr_eps0.01 0.001 -1 yes


python eval_roc_new.py 278 10,10,100,100,200,200 bb_raw no _ptetaflavloss_focalloss_gamma25.0,_flatptetaflavloss_focalloss_gamma25.0,_ptetaflavloss_focalloss_gamma25.0,_flatptetaflavloss_focalloss_gamma25.0,_ptetaflavloss_focalloss_gamma25.0,_flatptetaflavloss_focalloss_gamma25.0 0.001 -1 yes no -1 no ROC

```


## Plot inputs

several (1 to including 66)
```bash
python submit_eval_inputs.py -v 1,66 -a fgsm -r yes -pa '0.01,0.02' -f 278 -p 200 -w _ptetaflavloss_focalloss_gamma25.0_adv_tr_eps0.01 -d 0.001 -j -1 -me no
```
just 0 for example
```bash
python submit_eval_inputs.py -v 0,0 -a fgsm -r yes -pa '0.01,0.02' -f 278 -p 200 -w _ptetaflavloss_focalloss_gamma25.0_adv_tr_eps0.01 -d 0.001 -j -1 -me no
```
just 4 for example
```bash
python submit_eval_inputs.py -v 4,4 -a noise -r yes -pa '0.01,0.02' -f 278 -p 200 -w _ptetaflavloss_focalloss_gamma25.0 -d 0.001 -j -1 -me no
python submit_eval_inputs.py -v 4,4 -a fgsm -r yes -pa '0.01,0.02' -f 278 -p 200 -w _ptetaflavloss_focalloss_gamma25.0 -d 0.001 -j -1 -me no
python submit_eval_inputs.py -v 4,4 -a fgsm -r yes -pa '0.01,0.02' -f 278 -p 200 -w _ptetaflavloss_focalloss_gamma25.0_adv_tr_eps0.01 -d 0.001 -j -1 -me no
```