import os

# os.system("python node_feature_ml.py --min_block 6")
# os.system("python tsne.py")
# # os.system("python umap.py --model_name jmp-s-finetuned --normalize True")
# os.system("python umap.py --model_name jmp-l")
# os.system("python finetune_with_xyz.py --num_blocks 4 --batch_size 16 --xyz_path ./temp_data/LiOMn-mptrj.xyz --include_stress True --gpu 1")
# os.system("python finetune_with_xyz.py --num_blocks 2 --batch_size 8 --padding_method repeat --gpu 0,1")
# os.system("python finetune_with_xyz.py --num_blocks 1 --batch_size 64 --padding_method repeat")
import torch

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
