import matplotlib.pyplot as plt

def plot_train_valid(train_losses, valid_losses):
    if not (len(train_losses) == len(valid_losses)):
        print(f"Wrong lengths - could not plot, train: {len(train_losses)}, validation: {len(valid_losses)}")
        return
    x = [i for i in range(len(train_losses))]
    fig, ax = plt.subplots()
    ax.plot(x, train_losses, label='training loss')
    ax.plot(x, valid_losses, label='validation loss')
    legend = ax.legend(loc='upper right')
    plt.show()    