python main.py test -d datasets/cityscapes -s 512 --batch-size 1 --random-scale 2 --random-rotate 10 --epochs 200 --lr 0.007 --momentum 0.9 --lr-mode poly --workers 8 --list-dir datasets/cityscapes_lists -c 34 --resume checkpoint_latest.pth.tar --ms