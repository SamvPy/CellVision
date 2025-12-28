from matplotlib import pyplot as plt

def show_images(cells, x, y, size):
    fig, ax = plt.subplots(x, y, figsize=(size[0],size[1]))

    for ax_, cell in zip(ax.flatten(), cells):
        ax_.imshow(
            cell[0].squeeze(),
            cmap='grey',
        )
        ax_.set_title(cell[1])
    plt.show()