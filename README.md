# Ramen

Key packages
```
torch          2.4.1
torchvision    0.19.1
clip           1.0
```

To run experiments: 
```shell
cd ./shell
./run_ramen.sh
```
The GPU Memory Usage of Ramen for each setting: 
- ViT-B/32 on CIFAR10C: 11,564 MiB, 
- ViT-B/16 on CIFAR100C: 14,526 MiB, 
- ViT-L/14 on ImageNetC5K: 30,588 MiB, 
- ViT-B/32 on DomainNet: 11,568 MiB. 