import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_convergence_with_perturbation(seeds, perturb=False):
    """
    Plot convergence curves with mean and variability across multiple seeds,
    highlighting perturbation events if present.

    Parameters:
        seeds (list of int): List of seed values to iterate over.
        perturb (bool): Whether to use perturbed or regular results.
    """
    # Prepare file paths based on seeds and perturbation flag
    file_paths = []
    for seed in seeds:
        if perturb:
            file_paths.append(f"conv_results/perturbed/convergence_seed_{seed}.csv")
        else:
            file_paths.append(f"conv_results/regular/convergence_seed_{seed}.csv")

    # Initialize lists to store data from all files
    all_loss = []
    all_accuracy = []
    perturbation_events = []  # List to store perturbation rounds for each file

    # Read each file and store the data
    for file in file_paths:
        try:
            data = pd.read_csv(file, header=None, names=["loss", "accuracy", "perturbation_event"])

            if data.empty:
                print(f"File is empty: {file}")
                continue

            all_loss.append(data["loss"].values)
            all_accuracy.append(data["accuracy"].values)

            # Detect perturbation event (row where the third column is 1)
            perturbation_row = data[data["perturbation_event"] == 1].index
            if not perturbation_row.empty:
                perturbation_row = perturbation_row[0]
                perturbation_events.append((perturbation_row, data.iloc[perturbation_row]["accuracy"], data.iloc[perturbation_row]["loss"]))
            else:
                perturbation_events.append(None)
        except FileNotFoundError:
            print(f"File not found: {file}")
            continue
        except pd.errors.EmptyDataError:
            print(f"Empty data in file: {file}")
            continue

    # Ensure there is at least one valid file to process
    if len(all_loss) == 0 or len(all_accuracy) == 0:
        print("No valid data found. Exiting plot function.")
        return

    # Convert lists to arrays
    all_loss = np.array(all_loss)  # Shape: (num_seeds, num_epochs)
    all_accuracy = np.array(all_accuracy)  # Shape: (num_seeds, num_epochs)

    # Compute mean and standard deviation across seeds
    mean_loss = np.mean(all_loss, axis=0)
    std_loss = np.std(all_loss, axis=0)
    mean_accuracy = np.mean(all_accuracy, axis=0)
    std_accuracy = np.std(all_accuracy, axis=0)

    # Generate epochs range
    epochs = range(1, len(mean_loss) + 1)  # Epoch indices

    # Plot loss convergence curve with fill_between
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)  # Subplot 1: Loss
    print(perturbation_row)
    if perturb :
        plt.axvline(perturbation_row+1, linestyle='-.', color='red', label=r'CPA$^D$ perturbation' )
    plt.plot(epochs, mean_loss, label="Loss", marker="s", color="blue", linestyle="-")
    plt.fill_between(epochs, mean_loss - std_loss, mean_loss + std_loss, color="blue", alpha=0.2)
    for i, event in enumerate(perturbation_events):
        if event is not None:
            perturbation_epoch, _, loss_value = event
            #plt.scatter(perturbation_epoch + 1, loss_value, color="red", label=f"Perturbation (Seed {seeds[i]})")

    plt.xlabel("Rounds", fontsize=20)
    plt.ylabel("Loss", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True)
    plt.legend(fontsize=20)

    # Plot accuracy convergence curve with fill_between
    plt.subplot(1, 2, 2)  # Subplot 2: Accuracy
    if perturb :
        plt.axvline(perturbation_row+1, linestyle='-.', color='red', label=r'CPA$^D$ perturbation')
    plt.plot(epochs, mean_accuracy, label="Accuracy", marker="s", color="green", linestyle="-")
    plt.fill_between(epochs, mean_accuracy - std_accuracy, mean_accuracy + std_accuracy, color="green", alpha=0.2)
    for i, event in enumerate(perturbation_events):
        if event is not None:
            perturbation_epoch, accuracy_value, _ = event
            #plt.scatter(perturbation_epoch + 1, accuracy_value, color="red", label=f"Perturbation (Seed {seeds[i]})")

    plt.xlabel("Rounds", fontsize=20)
    plt.ylabel("Accuracy", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True)
    plt.legend(fontsize=20)

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()




def plot_convergence_comparison(seeds, iid=True, time='early'):
    """
    Plot convergence curves for both perturbed and non-perturbed cases, showing mean and variability.

    Parameters:
        seeds (list of int): List of seed values to iterate over.
    """
    # Prepare file paths for perturbed and non-perturbed cases
    if iid:
        non_perturbed_files = [f"conv_results/iid-data/regular/convergence_seed_{seed}.csv" for seed in seeds]
        if time == 'early' :
            print('early iid')
            perturbed_files = [f"conv_results/iid-data/perturbed/early/convergence_seed_{seed}.csv" for seed in seeds]
        elif time == 'late' :
            print('late iid')
            perturbed_files = [f"conv_results/iid-data/perturbed/late/convergence_seed_{seed}.csv" for seed in seeds]
        elif time == 'intermediate' :
            print('late iid')
            perturbed_files = [f"conv_results/iid-data/perturbed/intermediate/convergence_seed_{seed}.csv" for seed in seeds]

    else:
        non_perturbed_files = [f"conv_results/non-iid-data/regular/convergence_seed_{seed}.csv" for seed in seeds]
        if time == 'early':
            print('early non-iid')
            perturbed_files = [f"conv_results/non-iid-data/perturbed/early/convergence_seed_{seed}.csv" for seed in seeds]
        elif time == 'late' :
            print('late non-iid')
            perturbed_files = [f"conv_results/non-iid-data/perturbed/late/convergence_seed_{seed}.csv" for seed in seeds]
        elif time == 'intermediate' :
            print('late iid')
            perturbed_files = [f"conv_results/non-iid-data/perturbed/intermediate/convergence_seed_{seed}.csv" for seed in seeds]


    def process_files(file_paths):
        all_loss = []
        all_accuracy = []
        perturbation_events = []

        for file in file_paths:
            try:
                data = pd.read_csv(file, header=None, names=["loss", "accuracy", "perturbation_event"])

                if data.empty:
                    print(f"File is empty: {file}")
                    continue

                all_loss.append(data["loss"].values)
                all_accuracy.append(data["accuracy"].values)

                # Detect perturbation event
                perturbation_row = data[data["perturbation_event"] == 1].index
                if not perturbation_row.empty:
                    perturbation_row = perturbation_row[0]
                    perturbation_events.append((perturbation_row, data.iloc[perturbation_row]["accuracy"], data.iloc[perturbation_row]["loss"]))
                else:
                    perturbation_events.append(None)
            except FileNotFoundError:
                print(f"File not found: {file}")
                continue
            except pd.errors.EmptyDataError:
                print(f"Empty data in file: {file}")
                continue

        return np.array(all_loss), np.array(all_accuracy), perturbation_events

    # Process files for both cases
    perturbed_loss, perturbed_accuracy, perturbed_events = process_files(perturbed_files)
    non_perturbed_loss, non_perturbed_accuracy, non_perturbed_events = process_files(non_perturbed_files)

    # Ensure there is valid data for both cases
    if perturbed_loss.size == 0 or non_perturbed_loss.size == 0:
        print("No valid data found for one or both cases. Exiting plot function.")
        return

    # Compute mean and standard deviation
    def compute_stats(data):
        return np.mean(data, axis=0), np.std(data, axis=0)

    mean_perturbed_loss, std_perturbed_loss = compute_stats(perturbed_loss)
    mean_perturbed_accuracy, std_perturbed_accuracy = compute_stats(perturbed_accuracy)

    mean_non_perturbed_loss, std_non_perturbed_loss = compute_stats(non_perturbed_loss)
    mean_non_perturbed_accuracy, std_non_perturbed_accuracy = compute_stats(non_perturbed_accuracy)

    # Generate epochs range
    epochs = range(1, len(mean_perturbed_loss) + 1)

    # Create a figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot loss convergence curve
    ax1 = axes[0]
    ax1.plot(epochs, mean_perturbed_loss, label=r"CPA$^D$ Perturbed", color="blue", marker="s")
    ax1.fill_between(epochs, mean_perturbed_loss - std_perturbed_loss, mean_perturbed_loss + std_perturbed_loss, color="blue", alpha=0.2)

    ax1.plot(epochs, mean_non_perturbed_loss, label="Non-Perturbed", color="green", marker="^")
    ax1.fill_between(epochs, mean_non_perturbed_loss - std_non_perturbed_loss, mean_non_perturbed_loss + std_non_perturbed_loss, color="green", alpha=0.2)

    # Highlight perturbation events
    for i, event in enumerate(perturbed_events):
        if event is not None:
            perturbation_epoch, _, loss_value = event
            ax1.axvline(perturbation_epoch + 1, color="red", linestyle="--")

    ax1.set_xlabel("Rounds", fontsize=20)
    ax1.set_ylabel("Loss", fontsize=20)
    ax1.tick_params(axis="x", labelsize=20)
    ax1.tick_params(axis="y", labelsize=20)
    ax1.grid(True)

    # Plot accuracy convergence curve
    ax2 = axes[1]
    ax2.plot(epochs, mean_perturbed_accuracy, label=r"", color="blue", marker="s")
    ax2.fill_between(epochs, mean_perturbed_accuracy - std_perturbed_accuracy, mean_perturbed_accuracy + std_perturbed_accuracy, color="blue", alpha=0.2)

    ax2.plot(epochs, mean_non_perturbed_accuracy, label="", color="green", marker="^")
    ax2.fill_between(epochs, mean_non_perturbed_accuracy - std_non_perturbed_accuracy, mean_non_perturbed_accuracy + std_non_perturbed_accuracy, color="green", alpha=0.2)

    # Highlight perturbation events
    for i, event in enumerate(perturbed_events):
        if event is not None:
            perturbation_epoch, accuracy_value, _ = event
            if i == 0 :
                ax2.axvline(perturbation_epoch + 1, color="red", label=r"CPA$^D$ perturbation", linestyle="--")
            else :
                ax2.axvline(perturbation_epoch + 1, color="red", linestyle="--")
    ax2.set_xlabel("Rounds", fontsize=20)
    ax2.set_ylabel("Accuracy", fontsize=20)
    ax2.tick_params(axis="x", labelsize=20)
    ax2.tick_params(axis="y", labelsize=20)
    ax2.grid(True)

    # Add a shared legend for both subplots
    fig.legend(loc="upper center", ncol=3, fontsize=20, frameon=True, bbox_to_anchor=(0.53, 1.02))

    # Adjust layout and display the plot
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.show()




# Example usage
seeds = [0, 1, 2, 3, 4]
#plot_convergence_with_perturbation(seeds, perturb=False)

seeds = [0, 1, 2, 3, 4, 5]
#plot_convergence_with_perturbation(seeds, perturb=True)

#plot_convergence_with_perturbation(seeds, perturb=False)
plot_convergence_comparison(seeds, iid=True, time='early')
plot_convergence_comparison(seeds, iid=True, time='intermediate')
plot_convergence_comparison(seeds, iid=True, time='late')

#plot_convergence_comparison(seeds, iid=False, time='early')
#plot_convergence_comparison(seeds, iid=False, time='intermediate')
#plot_convergence_comparison(seeds, iid=False, time='late')
