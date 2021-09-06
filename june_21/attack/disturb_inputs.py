import numpy as np
import torch

def apply_noise(sample, magn=1e-2,offset=[0], default=0.001, dev="cpu"):
    scalers = [torch.load(f'/hpcwork/um106329/june_21/scaler_{i}_with_default_{default}.pt') for i in range(67)]
    with torch.no_grad():
        if default == int(default):
            default = int(default)
        minima = np.load('/home/um106329/aisafety/april_21/from_Nik/default_value_studies_minima.npy')
        defaults_per_variable = minima - default
        #device = torch.device("cpu")
        device = torch.device(dev)
        #scalers = torch.load(scalers_file_paths[s])
        #test_inputs =  torch.load(test_input_file_paths[s]).to(device).float()
        #val_inputs =  torch.load(val_input_file_paths[s]).to(device).float()
        #train_inputs =  torch.load(train_input_file_paths[s]).to(device).float()
        #test_targets =  torch.load(test_target_file_paths[s]).to(device)
        #val_targets =  torch.load(val_target_file_paths[s]).to(device)
        #train_targets =  torch.load(train_target_file_paths[s]).to(device)            
        #all_inputs = torch.cat((test_inputs,val_inputs,train_inputs))

        noise = torch.Tensor(np.random.normal(offset,magn,(len(sample),67))).to(device)
        xadv = sample + noise
        #all_inputs_noise = all_inputs + noise
        #xadv = scalers[variable].inverse_transform(all_inputs_noise[:,variable].cpu())
        integervars = [59,63,64,65,66]
        for variable in integervars:
            xadv[:,variable] = sample[:,variable]


        for i in range(67):
            if i in [41,42,43,44,45,46,47,49,50,51,52,53,54,55]:
                defaults = abs(scalers[i].inverse_transform(sample[:,i].cpu()) - defaults_per_variable[i]) < 2   # "floating point error" --> allow some error margin
            elif i in [19,20,21]:
                defaults = abs(scalers[i].inverse_transform(sample[:,i].cpu()) - defaults_per_variable[i]) < 0.4   # "floating point error" --> allow some error margin
            else:
                defaults = abs(scalers[i].inverse_transform(sample[:,i].cpu()) - defaults_per_variable[i]) < 0.001   # "floating point error" --> allow some error margin
            if np.sum(defaults) != 0:
                xadv[:,i][defaults] = sample[:,i][defaults]

        return xadv

def fgsm_attack(epsilon=1e-2,sample=None,targets=None,thismodel=None,thiscriterion=None,reduced=True, default=0.001, dev="cpu"):
    if default == int(default):
        default = int(default)
    minima = np.load('/home/um106329/aisafety/april_21/from_Nik/default_value_studies_minima.npy')
    defaults_per_variable = minima - default
    
    scalers = [torch.load(f'/hpcwork/um106329/june_21/scaler_{i}_with_default_{default}.pt') for i in range(67)]
    
    device = torch.device(dev)
    xadv = sample.clone().detach()
    
    # calculate the gradient of the model w.r.t. the *input* tensor:
    # first we tell torch that x should be included in grad computations
    xadv.requires_grad = True
    
    # from the undisturbed predictions, both the model and the criterion are already available and can be used here again; it's just that they were each part of a function, so not
    # automatically in the global scope
    if thismodel==None and thiscriterion==None:
        global model
        global criterion
    
    # then we just do the forward and backwards pass as usual:
    preds = thismodel(xadv)
    #print(targets)
    #print(torch.unique(targets))
    #print(preds)
    loss = thiscriterion(preds, targets).mean()
    # maybe add sample weights here as well for the ptetaflavloss weighting method
    thismodel.zero_grad()
    loss.backward()
    
    with torch.no_grad():
        #now we obtain the gradient of the input. It has the same dimensions as the tensor xadv, and it "points" in the direction of increasing loss values.
        dx = torch.sign(xadv.grad.detach())
        
        #so, we take a step in that direction!
        xadv += epsilon*torch.sign(dx)
        
        #remove the impact on selected variables. This is nessecary to avoid problems that occur otherwise in the input shapes.
        if reduced:
            integervars = [59,63,64,65,66]
            for variable in integervars:
                xadv[:,variable] = sample[:,variable]

            for i in range(67):
                if i in [41,42,43,44,45,46,47,49,50,51,52,53,54,55]:
                    defaults = abs(scalers[i].inverse_transform(sample[:,i].cpu()) - defaults_per_variable[i]) < 2   # "floating point error" --> allow some error margin
                elif i in [19,20,21]:
                    defaults = abs(scalers[i].inverse_transform(sample[:,i].cpu()) - defaults_per_variable[i]) < 0.4   # "floating point error" --> allow some error margin
                else:
                    defaults = abs(scalers[i].inverse_transform(sample[:,i].cpu()) - defaults_per_variable[i]) < 0.001   # "floating point error" --> allow some error margin
                if np.sum(defaults) != 0:
                    xadv[:,i][defaults] = sample[:,i][defaults]
        return xadv.detach()
