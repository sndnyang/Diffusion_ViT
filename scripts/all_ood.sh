
echo "px"

echo svhn
python eval_model.py --resume $1 --eval OOD --ood_dataset svhn --score_fn px --gpu-id $2

echo cifar interp
python eval_model.py --resume $1 --eval OOD --ood_dataset cifar_interp --score_fn px --gpu-id $2

echo cifar100
python eval_model.py --resume $1 --eval OOD --ood_dataset cifar_100 --score_fn px --gpu-id $2

echo celeba
python eval_model.py --resume $1 --eval OOD --ood_dataset celeba --score_fn px --gpu-id $2


echo "py"
echo svhn
python eval_model.py --resume $1 --eval OOD --ood_dataset svhn --score_fn py --gpu-id $2

echo cifar interp
python eval_model.py --resume $1 --eval OOD --ood_dataset cifar_interp --score_fn py --gpu-id $2

echo cifar100
python eval_model.py --resume $1 --eval OOD --ood_dataset cifar_100 --score_fn py --gpu-id $2

echo celeba
python eval_model.py --resume $1 --eval OOD --ood_dataset celeba --score_fn py --gpu-id $2
