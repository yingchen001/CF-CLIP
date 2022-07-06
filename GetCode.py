


import os
import pickle
import numpy as np
from dnnlib import tflib  
import tensorflow as tf 
import torch
import argparse

def LoadModel(model_path):
    # Initialize TensorFlow.
    tflib.init_tf()
    
    tmp=os.path.join(model_path)
    with open(tmp, 'rb') as f:
        _, _, Gs = pickle.load(f)
    return Gs

def lerp(a,b,t):
     return a + (b - a) * t


def GetCode_torch(Gs,random_state,num_img,num_once,post_fix):
    rnd = np.random.RandomState(random_state)  #5
    
    truncation_psi=0.7
    truncation_cutoff=8
    
    dlatent_avg=Gs.get_var('dlatent_avg')
    
    dlatents=np.zeros((num_img,512),dtype='float32')
    for i in range(int(num_img/num_once)):
        src_latents =  rnd.randn(num_once, Gs.input_shape[1])
        src_dlatents = Gs.components.mapping.run(src_latents, None) # [seed, layer, component]
        
        # Apply truncation trick.
        if truncation_psi is not None and truncation_cutoff is not None:
                layer_idx = np.arange(src_dlatents.shape[1])[np.newaxis, :, np.newaxis]
                ones = np.ones(layer_idx.shape, dtype=np.float32)
                coefs = np.where(layer_idx < truncation_cutoff, truncation_psi * ones, ones)
                src_dlatents_np=lerp(dlatent_avg, src_dlatents, coefs)
                src_dlatents=src_dlatents_np[:,0,:].astype('float32')
                dlatents[(i*num_once):((i+1)*num_once),:]=src_dlatents
    print('get all z and w')
    
    tmp='./latent_codes/'+'/W_'+post_fix+'.pt'
    dlatents = torch.from_numpy(dlatents)
    torch.save(dlatents,tmp)



#%%
if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser(description='Process some integers.')
    
    parser.add_argument('--model_path',type=str,
                    help='path to .pkl stylegan model')
    parser.add_argument('--dataset_name',type=str,
                    help='name of the dataset, eg. cat')
    
    args = parser.parse_args()
    random_state=5
    num_img=100_000 
    num_once=1_000
    dataset_name=args.dataset_name
    model_path = args.model_path

    Gs=LoadModel(model_path)
    GetCode_torch(Gs,random_state,num_img,num_once,dataset_name+'_train')
    GetCode_torch(Gs,random_state,3000,num_once,dataset_name+'_test')

    
    
    
    
    
