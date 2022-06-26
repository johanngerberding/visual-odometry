import matplotlib.pyplot as plt 


def plot_results(gt_path: list, estimated_path: list, title: str, save: str = ""):
    "Plot the estimated path and the ground truth path." 
    plt.figure(figsize=(16,9))
    plt.plot(
        [el[0] for el in gt_path], 
        [el[1] for el in gt_path], 
        'g', 
        label="Ground truth")
    plt.plot(
        [el[0] for el in estimated_path], 
        [el[1] for el in estimated_path], 
        'b', 
        label="Estimated path")
    plt.xlabel('x')
    plt.ylabel('z') 
    plt.title(title)
    plt.legend()
    plt.show()

    if save:
        assert save.endswith(".jpg") 
        plt.savefig(save) 