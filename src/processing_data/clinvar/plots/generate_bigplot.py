import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from PIL import Image


def fig_1(file_path_dir,image_order = [
        ['A.png', 'A2.png'],
        ['C.png', 'B.png'],
        ['C2.png', 'D.png'],
        ['E.png', 'F.png']
    ],resize_dim=(400, 400)):
    # Define the image order you want


    # Create the figure
    # If we want 1998x1949 pixels and use dpi=100, then:
    # width_in_inches = 1998 / 100 = 19.98
    # height_in_inches = 1949 / 100 = 19.49
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(19.98, 19.49), dpi=300)

    # Iterate over rows and columns to load and display images
    for r in range(4):
        for c in range(2):
            img_name = image_order[r][c]
            img_path = os.path.join(file_path_dir, img_name)
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image {img_path} not found in the current directory.")

            # Use PIL to open and resize the image
            with Image.open(img_path) as im:
                # Resize image to desired dimensions
                im_resized = im.resize(resize_dim, Image.LANCZOS)


            img = mpimg.imread(img_path)
            axes[r][c].imshow(img)
            axes[r][c].axis('off')  # Hide the axis for a cleaner look

    plt.tight_layout()
    plt.savefig(os.path.join(file_path_dir,"bigplot.png"), dpi=300)
    plt.show()

# If you want to save the figure:
# plt.savefig("bigplot.png", dpi=100)

if __name__ == '__main__':
    file_path_dir = '/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations/plots/fig2'
    first_image = [
        ['A.png', 'A2.png'],
        ['C.png', 'B.png'],
        ['C2.png', 'D.png'],
        ['E.png', 'F.png']
    ]
    fig_1(file_path_dir, first_image)
