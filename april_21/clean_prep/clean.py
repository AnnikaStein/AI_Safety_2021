import uproot4 as uproot
import numpy as np
import awkward1 as ak

import gc

import json


### Parser ###

import argparse
import ast


parser = argparse.ArgumentParser(description="Perform step 1 of data cleaning")
parser.add_argument("startfile", type=int, help="Number of the file with which cleaning starts")
parser.add_argument("endfile", type=int, help="Number of the file with with cleaning ends")
parser.add_argument("default", type=float, help="default value relative to the minimum of the distribution, with positive sign")
args = parser.parse_args()

startindex = args.startfile
endindex = args.endfile
default = args.default


### The original root-files ###

list_paths = []
with open('/home/um106329/aisafety/new_march_21/qcd_file_paths.json') as json_file:
    json_paths = json.load(json_file)
    '''
    # QCD files
    for entry in json_paths['QCD_HT300to500_TuneCP5_13TeV-madgraph-pythia8']:
        list_paths.append(entry)
    for entry in json_paths['QCD_HT500to700_TuneCP5_13TeV-madgraph-pythia8']:
        list_paths.append(entry)
    for entry in json_paths['QCD_HT700to1000_TuneCP5_13TeV-madgraph-pythia8']:
        list_paths.append(entry)
    for entry in json_paths['QCD_HT1000to1500_TuneCP5_13TeV-madgraph-pythia8']:
        list_paths.append(entry)
    for entry in json_paths['QCD_HT1500to2000_TuneCP5_13TeV-madgraph-pythia8']:
        list_paths.append(entry)
    for entry in json_paths['QCD_HT2000toInf_TuneCP5_13TeV-madgraph-pythia8']:
        list_paths.append(entry)
    '''  
    # TTtoSemileptonic files
    for entry in json_paths['TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8']:
        list_paths.append(entry)


##################### Selecting default values ####################
minima = np.load('/home/um106329/aisafety/april_21/from_Nik/default_value_studies_minima.npy') #np.zeros(100)for constant default of -999 #have to be determined externally, otherwise they would not be the same for every iteration
defaults = minima - default


### Perform cleaning and save files ###

##################### Inserted default [i] at any point where it is referenced
def cleandataset(f):
# the feature-names are the attributes or columns of interest, in this case: information about Jets
    feature_names = [k for k in f['Events'].keys() if  (('Jet_eta' == k) or ('Jet_pt' == k) or ('Jet_DeepCSV' in k))]
    # tagger output to compare with later and variables used to get the truth output
    feature_names.extend(('Jet_btagDeepB_b','Jet_btagDeepB_bb', 'Jet_btagDeepC','Jet_btagDeepL'))
    feature_names.extend(('Jet_nBHadrons', 'Jet_hadronFlavour'))
    
    
    # go through a specified number of events, and get the information (awkward-arrays) for the keys specified above
    for data in f['Events'].iterate(feature_names, step_size=f['Events'].num_entries, library='ak'):
        break
        
    
    # creating an array to store all the columns with their entries per jet, flatten per-event -> per-jet
    datacolumns = np.zeros((len(feature_names)+1, len(ak.flatten(data['Jet_pt'], axis=1))))
   

    for featureindex in range(len(feature_names)):
        a = ak.flatten(data[feature_names[featureindex]], axis=1) # flatten along first inside to get jets
        
        datacolumns[featureindex] = ak.to_numpy(a)


    nbhad = ak.to_numpy(ak.flatten(data['Jet_nBHadrons'], axis=1))
    hadflav = ak.to_numpy(ak.flatten(data['Jet_hadronFlavour'], axis=1))

    target_class = np.full_like(hadflav, 3)                                                      # udsg
    target_class = np.where(hadflav == 4, 2, target_class)                                       # c
    target_class = np.where(np.bitwise_and(hadflav == 5, nbhad > 1), 1, target_class)            # bb
    target_class = np.where(np.bitwise_and(hadflav == 5, nbhad <= 1), 0, target_class)           # b, lepb

   

    datacolumns[len(feature_names)] = ak.to_numpy(target_class) 

    datavectors = datacolumns.transpose()
    
    
    #print(i)
    for j in range(len(datavectors[0])):
        datavectors[datavectors[:, j] == np.nan]  = defaults[j]
        datavectors[datavectors[:, j] <= -np.inf] = defaults[j]
        datavectors[datavectors[:, j] >= np.inf]  = defaults[j]
        datavectors[datavectors[:, j] == -999]  = defaults[j]  # this one line is new and the reason for that is that there can be "original" -999 defaults in the inputs that should now also move into the new
                                                               # default bin, it was not necessary in my old clean_1_2.py code, because I could just leave them where they are, here they need to to be modified
    
    datavecak = ak.from_numpy(datavectors)
    
    #print(len(datavecak),"entries before cleaning step 1")
    
    datavecak = datavecak[datavecak[:, 67] >= 0.]
    datavecak = datavecak[datavecak[:, 67] <= 1.]
    datavecak = datavecak[datavecak[:, 68] >= 0.]
    datavecak = datavecak[datavecak[:, 68] <= 1.]
    datavecak = datavecak[datavecak[:, 69] >= 0.]
    datavecak = datavecak[datavecak[:, 69] <= 1.]
    datavecak = datavecak[datavecak[:, 70] >= 0.]
    datavecak = datavecak[datavecak[:, 70] <= 1.]

    

    # check jetNSelectedTracks, jetNSecondaryVertices > 0
    datavecak = datavecak[(datavecak[:, 63] > 0) | (datavecak[:, 64] > 0)]  # keep those where at least any of the two variables is > 0, they don't need to be > 0 simultaneously
    #print(len(datavecak),"entries after cleaning step 1")

    alldata = ak.to_numpy(datavecak)
    
        
    
    for track0_vars in [6,12,22,29,35,42,50]:
        alldata[:,track0_vars][alldata[:,64] <= 0] = defaults[track0_vars]
    for track0_1_vars in [7,13,23,30,36,43,51]:
        alldata[:,track0_1_vars][alldata[:,64] <= 1] = defaults[track0_1_vars]
    for track01_2_vars in [8,14,24,31,37,44,52]:
        alldata[:,track01_2_vars][alldata[:,64] <= 2] = defaults[track01_2_vars]
    for track012_3_vars in [9,15,25,32,38,45,53]:
        alldata[:,track012_3_vars][alldata[:,64] <= 3] = defaults[track012_3_vars]
    for track0123_4_vars in [10,16,26,33,39,46,54]:
        alldata[:,track0123_4_vars][alldata[:,64] <= 4] = defaults[track0123_4_vars]
    for track01234_5_vars in [11,17,27,34,40,47,55]:
        alldata[:,track01234_5_vars][alldata[:,64] <= 5] = defaults[track01234_5_vars]
    alldata[:,18][alldata[:,65] <= 0] = defaults[18]
    alldata[:,19][alldata[:,65] <= 1] = defaults[19]
    alldata[:,20][alldata[:,65] <= 2] = defaults[20]
    alldata[:,21][alldata[:,65] <= 3] = defaults[21]

    for AboveCharm_vars in [41,48,49,56]:
        alldata[:,AboveCharm_vars][alldata[:,AboveCharm_vars]==-1] = defaults[AboveCharm_vars] 
    
    
    datacls = [i for i in range(0,67)]
    datacls.append(73)
    dataset = alldata[:, datacls]
    
    DeepCSV_dataset = alldata[:, 67:71]
    
    return dataset, DeepCSV_dataset


for innerstart in np.arange(startindex, endindex, 50):
    if innerstart + 49 <= endindex:
        for i,n in enumerate(range(innerstart, innerstart + 50)):
            #print(n)
            if i == 0:
                dataset, DeepCSV_dataset = cleandataset(uproot.open(list_paths[n]))
            else:
                datasetNEW, DeepCSV_datasetNEW = cleandataset(uproot.open(list_paths[n]))
                dataset, DeepCSV_dataset = np.concatenate((dataset, datasetNEW)), np.concatenate((DeepCSV_dataset, DeepCSV_datasetNEW))
        
        np.save(f'/hpcwork/um106329/april_21/cleaned_TT/inputs_{innerstart}_to_{innerstart+49}_with_default_{default}.npy', dataset)  
        np.save(f'/hpcwork/um106329/april_21/cleaned_TT/deepcsv_{innerstart}_to_{innerstart+49}_with_default_{default}.npy', DeepCSV_dataset) 
    else:
        for i,n in enumerate(range(innerstart, endindex)):
            #print(n)
            if i == 0:
                dataset, DeepCSV_dataset = cleandataset(uproot.open(list_paths[n]))
            else:
                datasetNEW, DeepCSV_datasetNEW = cleandataset(uproot.open(list_paths[n]))
                dataset, DeepCSV_dataset = np.concatenate((dataset, datasetNEW)), np.concatenate((DeepCSV_dataset, DeepCSV_datasetNEW))
        
        np.save(f'/hpcwork/um106329/april_21/cleaned_TT/inputs_{innerstart}_to_{endindex}_with_default_{default}.npy', dataset)  
        np.save(f'/hpcwork/um106329/april_21/cleaned_TT/deepcsv_{innerstart}_to_{endindex}_with_default_{default}.npy', DeepCSV_dataset) 
    gc.collect()
    


