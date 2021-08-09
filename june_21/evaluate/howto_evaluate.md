Create ROC curves example
```python
python eval_roc_new.py 278 200 BvL_eps0.01,BvL_raw no _ptetaflavloss_focalloss_gamma25.0_adv_tr_eps0.01 0.001 -1 yes


python eval_roc_new.py 278 10,10,100,100,200,200 bb_raw no _ptetaflavloss_focalloss_gamma25.0,_flatptetaflavloss_focalloss_gamma25.0,_ptetaflavloss_focalloss_gamma25.0,_flatptetaflavloss_focalloss_gamma25.0,_ptetaflavloss_focalloss_gamma25.0,_flatptetaflavloss_focalloss_gamma25.0 0.001 -1 yes no -1 no ROC

```