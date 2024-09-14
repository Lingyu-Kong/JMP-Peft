import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.image as mpimg
import argparse
import os


def main(args_dict:dict):
    image_path = args_dict["image_path"]
    all_files = os.listdir(image_path)
    all_files = [file for file in all_files if file.endswith(".png")]
    species = list(set([file.split("_")[0] for file in all_files]))
    
    for specie in species:
        specie_files = [file for file in all_files if file.split("_")[0] == specie]
        specie_files = sorted(specie_files)
        fig, ax = plt.subplots()
        ims = []
        captions = []
        for i, file in enumerate(specie_files):
            num_block = i + 1
            assert f"block{num_block}" in file, f"Wrong order of files: {file}, should be block{num_block}"
            im = mpimg.imread(os.path.join(image_path, file))
            ims.append(im)
            captions.append(f"Node Features after Block {num_block}")
        def animate(i):
            ax.clear()
            ax.imshow(ims[i])
            ax.axis("off")
            ax.text(0.5, -0.1, captions[i], fontsize=12, ha='center', va='top', transform=ax.transAxes)
        ani = animation.FuncAnimation(fig, animate, frames=len(ims), interval=2000)
        file_name = specie_files[0].replace(f"block1_", "").replace(".png", "")
        ani.save(os.path.join(image_path, file_name + ".gif"), writer="imagemagick")
        plt.close(fig)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Generate GIF from images")
    parser.add_argument("--image_path", type=str, default="./MgSi-m3gnet-fsc-tsne-cosine", help="Path to the folder containing images")
    args = parser.parse_args()
    args_dict = vars(args)
    main(args_dict)