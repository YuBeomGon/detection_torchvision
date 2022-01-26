# torchvision_detection

COCO dataset training  
python -W ignore -m torch.distributed.launch --nproc_per_node=3 --master_addr=192.168.40.242 --master_port=50019 --use_env train.py --sync-bn | tee outputs/coco_swin2_retina.log  
torchrun --nproc_per_node=3 train.py --epochs 26 --lr-steps 16 22 --a

