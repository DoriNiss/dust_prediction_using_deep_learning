```python
# !jupyter nbconvert --output-dir='html' --to html draft_model_vit_pv_to_z500.ipynb
# !jupyter nbconvert --output-dir='script' --to markdown draft_model_vit_pv_to_z500.ipynb
```

    [NbConvertApp] Converting notebook draft_model_vit_pv_to_z500.ipynb to html
    [NbConvertApp] Writing 602605 bytes to html/draft_model_vit_pv_to_z500.html
    [NbConvertApp] Converting notebook draft_model_vit_pv_to_z500.ipynb to markdown
    [NbConvertApp] Writing 3651 bytes to script/draft_model_vit_pv_to_z500.md



```python
# !pip install timm
```


```python
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset,DataLoader
import timm.models.vision_transformer as ViT
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import default_collate 
from tqdm import tqdm

import sys
sys.path.insert(0, '../../packages/')
from models.ViTtoMap import *
# from training.train_model import *
# from utils.training_loop_plotting import *

```


```python
location = "wexac"

if location == "wexac":
    data_dir = "/home/labs/rudich/Collaboration/dust_prediction/data/pv_to_z500_wide/datasets/"

sample_input = torch.load(data_dir+"split1_valid_input.pkl")
sample_target = torch.load(data_dir+"split1_valid_target.pkl")

```


```python
sample_input.shape, sample_target.shape
```




    (torch.Size([17528, 7, 81, 169]), torch.Size([17528, 1, 81, 169]))




```python
x = sample_input[0:2]
x.shape
```




    torch.Size([2, 7, 81, 169])




```python
vit = ViT.VisionTransformer(img_size=(81,169), patch_size=(9,13), in_chans=7, num_classes=10, embed_dim=512, depth=8,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.)
```


```python
sample_output = vit(sample_input[0:2])
sample_output.shape
```




    torch.Size([2, 10])




```python
y = vit.forward_features(x)
```


```python
vit.patch_embed(x).shape
```




    torch.Size([2, 117, 512])




```python
x_vit = vit.patch_embed(x)
cls_token = vit.cls_token.expand(x_vit.shape[0], -1, -1) 
if vit.dist_token is None:
    x_vit = torch.cat((cls_token, x_vit), dim=1)
else:
    x_vit = torch.cat((cls_token, vit.dist_token.expand(x_vit.shape[0], -1, -1), x), dim=1)
x_vit[:,0,:]-cls_token[0,:,:]
```




    tensor([[0., 0., 0.,  ..., 0., 0., 0.],
            [0., 0., 0.,  ..., 0., 0., 0.]], grad_fn=<SubBackward0>)




```python
x_vit = vit.pos_drop(x_vit + vit.pos_embed)
x_vit = vit.blocks(x_vit)
x_vit = vit.norm(x_vit)
x_vit.shape
```




    torch.Size([2, 118, 512])




```python
vit.forward_features(x).shape
```




    torch.Size([2, 512])




```python
t = torch.rand([5,117,512])
t.shape
```




    torch.Size([5, 117, 512])




```python
t_reshaped = t.reshape(5,9,13,512)
t_reshaped.shape
```




    torch.Size([5, 9, 13, 512])




```python
t_transposed = t_reshaped.permute(0,3,1,2)
t_transposed.shape
```




    torch.Size([5, 512, 9, 13])




```python
512*9*13
```




    59904




```python
Hin = 9
Win = 13
```


```python
stride = [9,13]
padding = [0,0]
kernel_size = [9,13]
dilation = [1,1]
output_padding = [0,0]
```


```python
Hout = (Hin-1)*stride[0]-2*padding[0]+dilation[0]*(kernel_size[0]-1)+output_padding[0]+1
Wout = (Win-1)*stride[1]-2*padding[1]+dilation[1]*(kernel_size[1]-1)+output_padding[1]+1
Hout,Wout
```




    (81, 169)




```python
unembed = torch.nn.ConvTranspose2d(512, 1, [9,13], stride=[9,13], padding=0, output_padding=0, 
                                   groups=1, bias=True, dilation=1)
```


```python
unembed(t_transposed).shape
```




    torch.Size([5, 1, 81, 169])




```python
unembed.weight.shape
```




    torch.Size([512, 1, 9, 13])




```python

```


```python

```


```python

```


```python

```


```python

```


```python
vit_to_map = ViTtoMap(vit,[1,81,169])
```


```python
vit_to_map(x).shape
```




    torch.Size([2, 1, 81, 169])




```python

```


```python
import pandas as pd
timestamps = pd.to_datetime(["2000-01-01 00:00:00+0000'", "2000-02-01 00:00:00+0000'"])

df_numeric = pd.to_numeric(timestamps)
df_numeric[1], timestamps[1], pd.to_datetime(timestamps)[1]
```




    (949363200000000000,
     Timestamp('2000-02-01 00:00:00+0000', tz='UTC'),
     Timestamp('2000-02-01 00:00:00+0000', tz='UTC'))




```python
t = torch.Tensor(df_numeric).float()
pd.to_datetime(t)[1]
```




    Timestamp('2000-02-01 00:00:13.240107008')




```python

```


```python

```


```python

```


```python

```




    torch.Size([5, 1, 9, 6656])




```python
vit_decoder = ViT.VisionTransformer(img_size=(9, 6656), patch_size=(1,512), in_chans=1, num_classes=10, 
                                    embed_dim=117, depth=8, num_heads=9, mlp_ratio=1., qkv_bias=True, 
                                    representation_size=None, distilled=False, 
                                    drop_rate=0., attn_drop_rate=0., drop_path_rate=0.)
```


```python
t = torch.rand([5,1,9,13*512])
print(t.shape)
t = vit_decoder.patch_embed(t)
cls_token = vit_decoder.cls_token.expand(t.shape[0], -1, -1)
if vit_decoder.dist_token is None:
    t = torch.cat((cls_token, t), dim=1)
else:
    t = torch.cat((cls_token, vit_decoder.dist_token.expand(t.shape[0], -1, -1), t), dim=1)
t = vit_decoder.pos_drop(t + vit_decoder.pos_embed)
print(t.shape)
t = vit_decoder.blocks(t)
t = vit_decoder.norm(t)
print(t.shape)
```

    torch.Size([5, 1, 9, 6656])
    torch.Size([5, 118, 117])
    torch.Size([5, 118, 117])



```python

```
