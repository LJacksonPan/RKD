#!/usr/bin/env bash
# Teacher Network
python run.py --epochs 40 --lr_decay_epochs 25 30 35 --lr_decay_gamma 0.5 --batch 32 --base resnet50 --sample distance --margin 0.2 --embedding_size 512 --split_r 250 --save_dir cifar100_resnet50_512

python run.py --epochs 40 --lr_decay_epochs 25 30 35 --lr_decay_gamma 0.5 --batch 32 --base resnet50 --sample distance --margin 0.2 --embedding_size 512 --split_r 250 --save_dir cifar100_resnet50_512 --mode eval > txt_out/cifar100_resnet50_512_250k.txt


# Self-Distillation
python run_distill.py --epochs 80 --lr_decay_epochs 40 60 --lr_decay_gamma 0.1 --batch 32 --base resnet50 --embedding_size 512 --l2normalize false --dist_ratio 1 --angle_ratio 2 --teacher_base resnet50 --teacher_embedding_size 512 --split_r 250 --teacher_load cifar100_resnet50_512/best.pth --save_dir cifar100_student_resnet50_512_250k

python run_distill.py --epochs 80 --lr_decay_epochs 40 60 --lr_decay_gamma 0.1 --batch 32 --base resnet50 --embedding_size 512 --l2normalize false --dist_ratio 1 --angle_ratio 2 --teacher_base resnet50 --teacher_embedding_size 512 --split_r 250 --teacher_load cifar100_resnet50_512/best.pth --save_dir cifar100_student_resnet50_512_250k --eval true > txt_out/cifar100_resnet50_512_250k.txt

python run_distill.py --epochs 80 --lr_decay_epochs 40 60 --lr_decay_gamma 0.1 --batch 32 --base resnet50 --embedding_size 512 --l2normalize false --dist_ratio 1 --angle_ratio 2 --teacher_base resnet50 --teacher_embedding_size 512 --split_r 313 --teacher_load cifar100_resnet50_512/best.pth --save_dir cifar100_student_resnet50_512_313k

python run_distill.py --epochs 80 --lr_decay_epochs 40 60 --lr_decay_gamma 0.1 --batch 32 --base resnet50 --embedding_size 512 --l2normalize false --dist_ratio 1 --angle_ratio 2 --teacher_base resnet50 --teacher_embedding_size 512 --split_r 313 --teacher_load cifar100_resnet50_512/best.pth --save_dir cifar100_student_resnet50_512_313k --eval true > txt_out/cifar100_resnet50_512_313k.txt

python run_distill.py --epochs 80 --lr_decay_epochs 40 60 --lr_decay_gamma 0.1 --batch 32 --base resnet50 --embedding_size 512 --l2normalize false --dist_ratio 1 --angle_ratio 2 --teacher_base resnet50 --teacher_embedding_size 512 --split_r 376 --teacher_load cifar100_resnet50_512/best.pth --save_dir cifar100_student_resnet50_512_376k

python run_distill.py --epochs 80 --lr_decay_epochs 40 60 --lr_decay_gamma 0.1 --batch 32 --base resnet50 --embedding_size 512 --l2normalize false --dist_ratio 1 --angle_ratio 2 --teacher_base resnet50 --teacher_embedding_size 512 --split_r 376 --teacher_load cifar100_resnet50_512/best.pth --save_dir cifar100_student_resnet50_512_376k --eval true > txt_out/cifar100_resnet50_512_376k.txt

python run_distill.py --epochs 80 --lr_decay_epochs 40 60 --lr_decay_gamma 0.1 --batch 32 --base resnet50 --embedding_size 512 --l2normalize false --dist_ratio 1 --angle_ratio 2 --teacher_base resnet50 --teacher_embedding_size 512 --split_r 438 --teacher_load cifar100_resnet50_512/best.pth --save_dir cifar100_student_resnet50_512_438k

python run_distill.py --epochs 80 --lr_decay_epochs 40 60 --lr_decay_gamma 0.1 --batch 32 --base resnet50 --embedding_size 512 --l2normalize false --dist_ratio 1 --angle_ratio 2 --teacher_base resnet50 --teacher_embedding_size 512 --split_r 438 --teacher_load cifar100_resnet50_512/best.pth --save_dir cifar100_student_resnet50_512_438k --eval true > txt_out/cifar100_resnet50_512_438k.txt

python run_distill.py --epochs 80 --lr_decay_epochs 40 60 --lr_decay_gamma 0.1 --batch 32 --base resnet50 --embedding_size 512 --l2normalize false --dist_ratio 1 --angle_ratio 2 --teacher_base resnet50 --teacher_embedding_size 512 --split_r 500 --teacher_load cifar100_resnet50_512/best.pth --save_dir cifar100_student_resnet50_512_500k

python run_distill.py --epochs 80 --lr_decay_epochs 40 60 --lr_decay_gamma 0.1 --batch 32 --base resnet50 --embedding_size 512 --l2normalize false --dist_ratio 1 --angle_ratio 2 --teacher_base resnet50 --teacher_embedding_size 512 --split_r 500 --teacher_load cifar100_resnet50_512/best.pth --save_dir cifar100_student_resnet50_512_500k --eval true > txt_out/cifar100_resnet50_512_500k.txt



python run_distill.py --epochs 80 --lr_decay_epochs 40 60 --lr_decay_gamma 0.1 --batch 32 --base resnet50 --embedding_size 512 --l2normalize false --dist_ratio 1 --angle_ratio 2 --teacher_base resnet50 --teacher_embedding_size 512 --split_l 125 --split_r 375 --teacher_load cifar100_resnet50_512/best.pth --save_dir cifar100_student_resnet50_512_U01

python run_distill.py --epochs 80 --lr_decay_epochs 40 60 --lr_decay_gamma 0.1 --batch 32 --base resnet50 --embedding_size 512 --l2normalize false --dist_ratio 1 --angle_ratio 2 --teacher_base resnet50 --teacher_embedding_size 512 --split_l 125 --split_r 375 --teacher_load cifar100_resnet50_512/best.pth --save_dir cifar100_student_resnet50_512_U01 --eval true > txt_out/cifar100_resnet50_512_U01.txt

python run_distill.py --epochs 80 --lr_decay_epochs 40 60 --lr_decay_gamma 0.1 --batch 32 --base resnet50 --embedding_size 512 --l2normalize false --dist_ratio 1 --angle_ratio 2 --teacher_base resnet50 --teacher_embedding_size 512 --split_l 250 --split_r 500 --teacher_load cifar100_resnet50_512/best.pth --save_dir cifar100_student_resnet50_512_U1

python run_distill.py --epochs 80 --lr_decay_epochs 40 60 --lr_decay_gamma 0.1 --batch 32 --base resnet50 --embedding_size 512 --l2normalize false --dist_ratio 1 --angle_ratio 2 --teacher_base resnet50 --teacher_embedding_size 512 --split_l 250 --split_r 500 --teacher_load cifar100_resnet50_512/best.pth --save_dir cifar100_student_resnet50_512_U1 --eval true > txt_out/cifar100_resnet50_512_U1.txt





python run.py --epochs 40 --lr_decay_epochs 25 30 35 --lr_decay_gamma 0.5 --batch 32 --base resnet50 --sample distance --margin 0.2 --embedding_size 512 --split_r 313 --save_dir cifar100_resnet50_512_313k

python run.py --epochs 40 --lr_decay_epochs 25 30 35 --lr_decay_gamma 0.5 --batch 32 --base resnet50 --sample distance --margin 0.2 --embedding_size 512 --split_r 313 --save_dir cifar100_resnet50_512_313k --mode eval > txt_out/cifar100_resnet50_512_313k.txt

python run.py --epochs 40 --lr_decay_epochs 25 30 35 --lr_decay_gamma 0.5 --batch 32 --base resnet50 --sample distance --margin 0.2 --embedding_size 512 --split_r 376 --save_dir cifar100_resnet50_512_376k

python run.py --epochs 40 --lr_decay_epochs 25 30 35 --lr_decay_gamma 0.5 --batch 32 --base resnet50 --sample distance --margin 0.2 --embedding_size 512 --split_r 376 --save_dir cifar100_resnet50_512_376k --mode eval > txt_out/cifar100_resnet50_512_376k.txt

python run.py --epochs 40 --lr_decay_epochs 25 30 35 --lr_decay_gamma 0.5 --batch 32 --base resnet50 --sample distance --margin 0.2 --embedding_size 512 --split_r 438 --save_dir cifar100_resnet50_512_438k

python run.py --epochs 40 --lr_decay_epochs 25 30 35 --lr_decay_gamma 0.5 --batch 32 --base resnet50 --sample distance --margin 0.2 --embedding_size 512 --split_r 438 --save_dir cifar100_resnet50_512_438k --mode eval > txt_out/cifar100_resnet50_512_438k.txt

python run.py --epochs 40 --lr_decay_epochs 25 30 35 --lr_decay_gamma 0.5 --batch 32 --base resnet50 --sample distance --margin 0.2 --embedding_size 512 --split_r 500 --save_dir cifar100_resnet50_512_500k

python run.py --epochs 40 --lr_decay_epochs 25 30 35 --lr_decay_gamma 0.5 --batch 32 --base resnet50 --sample distance --margin 0.2 --embedding_size 512 --split_r 500 --save_dir cifar100_resnet50_512_500k --mode eval > txt_out/cifar100_resnet50_512_500k.txt