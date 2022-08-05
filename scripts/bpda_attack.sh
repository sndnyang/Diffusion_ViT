
for i in {1,2,4,8,12,16,22,30}
do
  echo $i
  CUDA_VISIBLE_DEVICES=0 python bpda_eot_attack.py $1 l_inf $i
done

for i in {50,100,150,200,250,300,350,400,450,500}
do
  echo $i
  CUDA_VISIBLE_DEVICES=0 python bpda_eot_attack.py $1 l_2 $i
done
