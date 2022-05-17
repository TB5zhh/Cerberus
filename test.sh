CUDA_VISIBLE_DEVICES=1 python -m pdb main.py test \
-d /mnt/yuhang/dataset/nyud2 \
--img-dir /mnt/yuhang/dataset/nyud2/000000 \
--classes 40 \
-s 512 \
--resume model_best.pth.tar \
--phase test \
--batch-size 1 \
--ms \
--workers 10
