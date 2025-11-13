cd ../src || exit

gpu=0

tta_algo='Ramen'
tta_mode='mixed'

datasets=('CIFAR10C' 'CIFAR100C' 'ImageNet5K' 'DomainNet')
models=('clip_vitbase32' 'clip_vitbase16' 'clip_vitlarge14' 'clip_vitbase32')
batchsizes=(100 100 50 100)

for i in {0..3}; do
  {
    CUDA_VISIBLE_DEVICES=${gpu} python main.py \
      --dataset ${datasets[i]} \
      --model ${models[i]} \
      --tta_algo ${tta_algo} \
      --tta_mode ${tta_mode} \
      --cuda \
      --batch_size ${batch_sizes[i]} \
      --save_to "../log/${datasets[i]}_${tta_mode}/${tta_algo}.csv"
  }
done