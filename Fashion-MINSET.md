```python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('datasets'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
```


```python
from fastai.vision import *
from PIL import Image
import os

```


    ---------------------------------------------------------------------------

    ModuleNotFoundError                       Traceback (most recent call last)

    <ipython-input-40-1adb0878d7f8> in <module>
    ----> 1 from fastai.vision import *
          2 from PIL import Image
          3 import os
    

    ModuleNotFoundError: No module named 'fastai'



```python
path = 'datasets'
mnist_train = pd.read_csv(path+'/fashion-mnist_train.csv')
mnist_test = pd.read_csv(path+'/fashion-mnist_test.csv')

# mnist_train's head:
mnist_train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label</th>
      <th>pixel1</th>
      <th>pixel2</th>
      <th>pixel3</th>
      <th>pixel4</th>
      <th>pixel5</th>
      <th>pixel6</th>
      <th>pixel7</th>
      <th>pixel8</th>
      <th>pixel9</th>
      <th>...</th>
      <th>pixel775</th>
      <th>pixel776</th>
      <th>pixel777</th>
      <th>pixel778</th>
      <th>pixel779</th>
      <th>pixel780</th>
      <th>pixel781</th>
      <th>pixel782</th>
      <th>pixel783</th>
      <th>pixel784</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>30</td>
      <td>43</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 785 columns</p>
</div>




```python
mkdir datasets\working\test
```


```python
mkdir datasets\working\train


```


```python
def csv2img(csv,path='datasets/working/train/'): 
    """
    Convert pixel values from .csv to .png image
    """
    for i in range(len(csv)):
        # csv.iloc[i,1:].to_numpy() returns pixel values array for i'th imag excluding the label 
        # next step: reshape the array to original shape(28,28) and add missing color channels 
        result = Image.fromarray(np.uint8(np.stack(np.rot90(csv.iloc[i,1:].to_numpy().reshape((28,28)))*3,axis=-1))) 
        # save the image:
        result.save(f'{path}{str(i)}.png')
        
    print(f'{len(csv)} images were created.')

# let's run the fuction:
csv2img(mnist_train)
csv2img(mnist_test,path='datasets/working/test/')
```

    60000 images were created.
    10000 images were created.
    


```python
len(os.listdir('datasets/working/train')) == len(mnist_train)
```




    True




```python
len(os.listdir('datasets/working/test')) == len(mnist_test)
```




    True




```python
dict_fashion = {
0:'T-shirt/top',
1:'Trouser',
2:'Pullover',
3:'Dress',
4:'Coat',
5:'Sandal',
6:'Shirt',
7:'Sneaker',
8:'Bag',
9:'Ankle boot'}

mnist_train['label_text'] = mnist_train['label'].apply(lambda x: dict_fashion[x])
mnist_test['label_text'] = mnist_test['label'].apply(lambda x: dict_fashion[x])

# add image names:
mnist_train['img'] = pd.Series([str(i)+'.png' for i in range(len(mnist_train))])
mnist_test['img'] = pd.Series([str(i)+'.png' for i in range(len(mnist_test))])
```


```python
mnist_train.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label</th>
      <th>pixel1</th>
      <th>pixel2</th>
      <th>pixel3</th>
      <th>pixel4</th>
      <th>pixel5</th>
      <th>pixel6</th>
      <th>pixel7</th>
      <th>pixel8</th>
      <th>pixel9</th>
      <th>...</th>
      <th>pixel777</th>
      <th>pixel778</th>
      <th>pixel779</th>
      <th>pixel780</th>
      <th>pixel781</th>
      <th>pixel782</th>
      <th>pixel783</th>
      <th>pixel784</th>
      <th>label_text</th>
      <th>img</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Pullover</td>
      <td>0.png</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Ankle boot</td>
      <td>1.png</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>30</td>
      <td>43</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Shirt</td>
      <td>2.png</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>T-shirt/top</td>
      <td>3.png</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Dress</td>
      <td>4.png</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 787 columns</p>
</div>




```python
mnist_train[['img','label_text']].to_csv('datasets/working/labels.csv',index=False)
mnist_test[['img','label_text']].to_csv('datasets/working/test.csv',index=False)
```


```python
data = (ImageList.from_csv('/kaggle/working/', 'labels.csv', folder='train')
        #Where to find the data? -> in '/kaggle/working/train' folder
        .split_by_rand_pct(seed=12)
        #How to split in train/valid? -> randomly with the default 20% in valid. There's an option to split by folfder or by id
        .label_from_df()
        #How to label? -> use the second column of the csv file and split the tags by ' '. / can be labeled by subfolder name/ can be labeled by applying regex to image name
        #.transform(tfms)
        #Data augmentation? -> use tfms with a size of 28. 
        .databunch() # change batch size and number of workers by passing arguments: (bs=32, num_workers=4, collate_fn=bb_pad_collate)
        #Finally -> use the defaults for conversion to databunch
        #.normalize()
       )   
        # Normalize x with mean and std, If you're using a pretrained model, you'll need to use the normalization that was used to train the model (e.g., imagenet_stats)

# Show image batch:
data.show_batch(rows=3, figsize=(4,4))
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-39-3c47d3de5c6d> in <module>
    ----> 1 data = (ImageList.from_csv('/kaggle/working/', 'labels.csv', folder='train')
          2         #Where to find the data? -> in '/kaggle/working/train' folder
          3         .split_by_rand_pct(seed=12)
          4         #How to split in train/valid? -> randomly with the default 20% in valid. There's an option to split by folfder or by id
          5         .label_from_df()
    

    NameError: name 'ImageList' is not defined



```python
data.show_batch(rows=3, figsize=(4,4))
```


![png](output_12_0.png)



```python
learn = cnn_learner(data, models.resnet50, metrics=[accuracy])
```

    Downloading: "https://download.pytorch.org/models/resnet50-19c8e357.pth" to /root/.cache/torch/checkpoints/resnet50-19c8e357.pth
    


    HBox(children=(FloatProgress(value=0.0, max=102502400.0), HTML(value='')))


    
    


```python
learn.fit_one_cycle(4)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.941544</td>
      <td>0.780751</td>
      <td>0.713750</td>
      <td>00:49</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.652636</td>
      <td>0.553753</td>
      <td>0.788000</td>
      <td>00:44</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.548311</td>
      <td>0.486505</td>
      <td>0.814333</td>
      <td>00:44</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.486254</td>
      <td>0.464214</td>
      <td>0.827417</td>
      <td>00:44</td>
    </tr>
  </tbody>
</table>



```python
def interpret_res(plot_top_losses=True, conf_matrix=True, most_confused=False):
    """
    Result interpretation includes top losses, confusion matrix, and most confused.
    """
    
    # plot top losses:
    if plot_top_losses==True:
        interp = ClassificationInterpretation.from_learner(learn)
        losses,idxs = interp.top_losses()
        len(data.valid_ds)==len(losses)==len(idxs)
        interp.plot_top_losses(9, figsize=(12,12))

    # plot confusion matrix:
    if conf_matrix==True:
        doc(interp.plot_top_losses)
        interp.plot_confusion_matrix(figsize=(12,12), dpi=60)

    # most confused:
    if most_confused==True:
        interp.most_confused(min_val=100)
```


```python
interpret_res()
```






![png](output_16_1.png)



![png](output_16_2.png)



```python
learn.lr_find()
learn.recorder.plot()
```



    <div>
        <style>
            /* Turns off some styling */
            progress {
                /* gets rid of default border in Firefox and Opera. */
                border: none;
                /* Needs to be in here for Safari polyfill so background images work as expected. */
                background-size: auto;
            }
            .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
                background: #F44336;
            }
        </style>
      <progress value='0' class='' max='1' style='width:300px; height:20px; vertical-align: middle;'></progress>
      0.00% [0/1 00:00<00:00]
    </div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table><p>

    <div>
        <style>
            /* Turns off some styling */
            progress {
                /* gets rid of default border in Firefox and Opera. */
                border: none;
                /* Needs to be in here for Safari polyfill so background images work as expected. */
                background-size: auto;
            }
            .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
                background: #F44336;
            }
        </style>
      <progress value='84' class='' max='750' style='width:300px; height:20px; vertical-align: middle;'></progress>
      11.20% [84/750 00:04<00:34 1.4446]
    </div>



    LR Finder is complete, type {learner_name}.recorder.plot() to see the graph.
    


![png](output_17_2.png)



```python
learn.unfreeze()
learn.fit_one_cycle(2, max_lr=slice(1e-5,1e-3))
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.434152</td>
      <td>0.368083</td>
      <td>0.859083</td>
      <td>00:51</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.302321</td>
      <td>0.323187</td>
      <td>0.878583</td>
      <td>00:51</td>
    </tr>
  </tbody>
</table>



```python
learn.save('learn_resnet50_stage_1')
```


```python
learn.lr_find()
learn.recorder.plot()
```



    <div>
        <style>
            /* Turns off some styling */
            progress {
                /* gets rid of default border in Firefox and Opera. */
                border: none;
                /* Needs to be in here for Safari polyfill so background images work as expected. */
                background-size: auto;
            }
            .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
                background: #F44336;
            }
        </style>
      <progress value='0' class='' max='1' style='width:300px; height:20px; vertical-align: middle;'></progress>
      0.00% [0/1 00:00<00:00]
    </div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table><p>

    <div>
        <style>
            /* Turns off some styling */
            progress {
                /* gets rid of default border in Firefox and Opera. */
                border: none;
                /* Needs to be in here for Safari polyfill so background images work as expected. */
                background-size: auto;
            }
            .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
                background: #F44336;
            }
        </style>
      <progress value='73' class='' max='750' style='width:300px; height:20px; vertical-align: middle;'></progress>
      9.73% [73/750 00:04<00:41 0.7176]
    </div>



    LR Finder is complete, type {learner_name}.recorder.plot() to see the graph.
    


![png](output_20_2.png)



```python
learn.fit_one_cycle(4, max_lr=slice(1e-6,1e-5))
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.250525</td>
      <td>0.322249</td>
      <td>0.879583</td>
      <td>00:53</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.236386</td>
      <td>0.323772</td>
      <td>0.880583</td>
      <td>00:51</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.221893</td>
      <td>0.326379</td>
      <td>0.880833</td>
      <td>00:51</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.221948</td>
      <td>0.328941</td>
      <td>0.880500</td>
      <td>00:51</td>
    </tr>
  </tbody>
</table>



```python
learn.save('learn_resnet50_stage_1')
```


```python
learn.lr_find()
learn.recorder.plot()
```



    <div>
        <style>
            /* Turns off some styling */
            progress {
                /* gets rid of default border in Firefox and Opera. */
                border: none;
                /* Needs to be in here for Safari polyfill so background images work as expected. */
                background-size: auto;
            }
            .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
                background: #F44336;
            }
        </style>
      <progress value='0' class='' max='1' style='width:300px; height:20px; vertical-align: middle;'></progress>
      0.00% [0/1 00:00<00:00]
    </div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table><p>

    <div>
        <style>
            /* Turns off some styling */
            progress {
                /* gets rid of default border in Firefox and Opera. */
                border: none;
                /* Needs to be in here for Safari polyfill so background images work as expected. */
                background-size: auto;
            }
            .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
                background: #F44336;
            }
        </style>
      <progress value='75' class='' max='750' style='width:300px; height:20px; vertical-align: middle;'></progress>
      10.00% [75/750 00:04<00:42 0.7217]
    </div>



    LR Finder is complete, type {learner_name}.recorder.plot() to see the graph.
    


![png](output_23_2.png)



```python
learn.fit_one_cycle(2, max_lr=slice(1e-5,1e-4))
```
