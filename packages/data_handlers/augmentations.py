import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate 
import numpy as np
from perlin_numpy import generate_perlin_noise_2d # use !pip install git+https://github.com/pvigier/perlin-numpy

class DustAugmentation:
    def __init__(self):
        pass

    def augment(self,meteorology_batch, dust_batch):
        return


class NoAugmentation(DustAugmentation):
    def __init__(self):
        pass

    def augment(self,meteorology_batch, dust_batch):
        return meteorology_batch


class PerlinAugmentation(DustAugmentation):
    """
        noise_tensor is the full tensor of which noise will be taken from, size: [N_clear+N_events,C,H,W]
        dust has the same indices as noise_tensor, used for labeling
        noise_odds: there is a chance of noise_odds to add any noise (the rest will stay untouched)
        Expecting tensors to be on cpu, use that before moving them to device
    """
    def __init__(self, noise_tensor, dust_tensor, dust_idx=0, th=73.4, noise_odds=0.8, perlin_size=(3,3), perlin_amplitude=0.1, debug=False):
        self.noise_odds = noise_odds
        self.perlin_size = perlin_size
        self.perlin_amplitude = perlin_amplitude
        self.dust_idx = dust_idx
        self.th = th
        self.debug = debug
        self.init_noise_tensors(noise_tensor, dust_tensor)

    def init_noise_tensors(self, noise_tensor, dust_tensor):
        self.noise_tensor_clear = noise_tensor[dust_tensor[:,self.dust_idx]<self.th]
        self.noise_tensor_events = noise_tensor[dust_tensor[:,self.dust_idx]>=self.th]
        
    def augment(self,meteorology_batch, dust_batch):
        # To be used after collating items into a batch (e.g. inside training loop)
        def augment_from_noise_tensor(initial_tensor, idxs_to_change, noise_tensor):
            _,C,H,W = initial_tensor.shape
            B = idxs_to_change.shape[0]
            N = noise_tensor.shape[0]
            num_zeros = int(N/self.noise_odds-N)
            noise_tensor_zeros_expanded = torch.cat((noise_tensor.new_zeros(num_zeros,C,H,W),noise_tensor))
            idxs_to_choose_from = torch.arange(noise_tensor_zeros_expanded.shape[0])*1.
            try:
                choosen_idxs = torch.multinomial(idxs_to_choose_from,num_samples=B,replacement=True)
                noise_choosen = noise_tensor_zeros_expanded[choosen_idxs].float()#.to(initial_tensor.device)
                perlin_noise = generate_perlin_noise_2d((H, W), self.perlin_size) # one for all clear or all events of batch
#                 perlin_noise_tensor = torch.tensor(perlin_noise,device=initial_tensor.device).expand(noise_choosen.shape[0],C,-1,-1).float()
                perlin_noise_tensor = (torch.ones_like(noise_choosen)*self.perlin_amplitude*perlin_noise).float()
#                 initial_tensor[idxs_to_change] = self.perlin_amplitude*perlin_noise_tensor+(1-self.perlin_amplitude)*noise_choosen               
                initial_tensor[idxs_to_change] = perlin_noise_tensor+(1-self.perlin_amplitude)*noise_choosen
            except Exception as exc:
                if self.debug:
                    print("Could not create augmented noisy tensors:")
                    print(exc)
                    print("idxs_to_choose_from:",idxs_to_choose_from,",num_samples:",B, 
                        "noise_tensor_zeros_expanded:",noise_tensor_zeros_expanded.shape, "idxs_to_change:",idxs_to_change,
                        idxs_to_change.shape, "initial_tensor[idxs_to_change]:",initial_tensor[idxs_to_change].shape)
            return initial_tensor        
        try:
            idxs_to_change_clear = (dust_batch[:,self.dust_idx]<self.th).nonzero()[:,0]
            idxs_to_change_events = (dust_batch[:,self.dust_idx]>=self.th).nonzero()[:,0]
            meteorology_batch = augment_from_noise_tensor(meteorology_batch, idxs_to_change_clear, self.noise_tensor_clear)
            meteorology_batch = augment_from_noise_tensor(meteorology_batch, idxs_to_change_events, self.noise_tensor_events)
        except Exception as exc:
            if self.debug:
                print("Could not create augmented noisy tensors - could not find clear or events in batch:")
                print(exc)
                print("dust:",dust_batch)
                print("idxs_to_change_clear:",(dust_batch[:,self.dust_idx]<self.th).nonzero(), 
                      "idxs_to_change_events:",(dust_batch[:,self.dust_idx]>=self.th).nonzero())
        return meteorology_batch            
