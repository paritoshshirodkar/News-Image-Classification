# News-Image-Classification

## Project Dependencies
```
python-3.6.2
cuda-8.0
cudnn-5.1
openblas-0.2.19
magma-2.3.0
pytorch-0.4.0_python-3.6.2
```

### Instructions to replicate baseline using Shared Computing Cluster

#### Submit an interactive rsh session
```
qrsh -P <PROJECT_NAME> -l gpus=<NUMBER_OF_GPUS> -l gpu_c=<GPU_CAPABILITY> -l h_rt=<HARD_TIME_LIMIT>
```
##### eg.
```
qrsh -P cs640grp -l gpus=1 -l gpu_c=4 -l h_rt=12:00:00
```

##### Load the dependency modules using
```
module load python/3.6.2
module load cuda/8.0
module load cudnn/5.1
module load openblas/0.2.19
module load magma/2.3.0
module load pytorch/0.4.0_python-3.6.2
```

##### Download the dataset that was provided to us
```
https://cs-people.bu.edu/yizheng/CS640_project/news_imgs.zip
```

##### unzip the files in the same directory that has the above .py files 


##### Start training using
```
python Train.py --cuda
```

##### Make predictions using
```
python Pred.py --image_dir test --model model_best.pth.tar --output_csvpath result.csv
```

##### Visualize the results using
```
python ResultVisualization.py
```

### Instructions to improve baseline using Shared Computing Cluster

#### We found the model gives the best accuracy by changing the learning rate to 0.001

##### Start training using
```
python Train.py --cuda --lr 0.001
```

##### Make predictions using
```
python Pred.py --image_dir test --model model_best.pth.tar --output_csvpath result.csv
```


##### Visualize the new results using
```
python ResultVisualization.py
```

##### [We have summarized the results for different improvements techiques over here](https://github.com/paritoshshirodkar/News-Image-Classification/tree/master/Results)


##### Pretrained models for different modifications can be found here:
https://drive.google.com/drive/folders/1B3BxgyudFfHQcLir381gCU3gULBpfAJA?usp=sharing

