from algorithms.algorithms import splaynet, lazy_splaynet, sliding_window_splaynet
from topology.CommunicationRequest import CommunicationRequest
from topology.SplayNetwork import SplayNetwork
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import random
import copy
import os


def csv_to_sequence(path):
    data = pd.read_csv(path, usecols=["src", "dst"])
    communication_sequence = [CommunicationRequest(i, row[1], row[2]) for i, row in enumerate(data.itertuples(), 1)]
    all_nodes = pd.concat([data["src"], data["dst"]]).unique()
    # nodes = [Node(i) for i in all_nodes] # dieser Schritt erstellt alle Node Instanzen
    return communication_sequence, all_nodes


def generate_temporal_sequence(nodes, tau, length):
    pairs = []
    prev_pair = None
    for _ in range(length):
        if prev_pair is not None and random.random() < tau:
            u, v = prev_pair
        else:
            u, v = random.sample(nodes, 2)
            while u == v:
                v = random.choice(nodes)
        pairs.append((u, v))
        prev_pair = (u, v)
    df = pd.DataFrame(pairs, columns=["src", "dst"])
    # df["size"] = 1  # nicht gebraucht, aber damit dfs einheitlich sind
    return df


def create_scatter_plot(csv_filename, output_folder, timestamp):
    """Generates a scatter plot for a given CSV file."""
    df = pd.read_csv(csv_filename)

    # Beispielhaftes Scatterplot mit zwei Spalten, passe die Namen entsprechend an
    if 'node' in df.columns and 'timestamp' in df.columns:
        plt.figure(figsize=(10, 6))
        plt.scatter(df['timestamp'], df['node'], alpha=0.6, marker='o')
        plt.xlabel('Timestamp')
        plt.ylabel('Node')
        plt.title(f'Scatter Plot for {os.path.basename(csv_filename)}')

        plot_filename = os.path.join(output_folder, f"scatter_{os.path.basename(csv_filename)[:-4]}_{timestamp}.png")
        plt.savefig(plot_filename)
        plt.close()
        print(f"Scatter plot saved at: {plot_filename}")
    else:
        print(f"Could not create scatter plot for {csv_filename}, missing required columns.")


def main():
    # Infos for plotting
    log_scale = False  # True if results should be plotted with logarithmic scale
    average = True  # True if average cost should be plotted

    combined_data = []
    sizes = []

    folderpath = "data"
    files = os.listdir(folderpath)

    for file in files:
        if file.endswith(".csv"):
            filepath = os.path.join(folderpath, file)
            sigma, nodes = csv_to_sequence(filepath)
            network_size = len(nodes)

            # Initialize tree network
            network = SplayNetwork()
            network.insert_balanced_BST(nodes)

            print(f"Processing dataset: {file} with {network_size} nodes.")
            # Apply algorithms, add loop for parameter finetuning here later
            print("Running SplayNet algorithm...")
            _, total_cost_splaynet = splaynet(copy.deepcopy(network), sigma)
            print(f"Finished with total cost {total_cost_splaynet}")
            print("Running Lazy SplayNet algorithm...")
            _, total_cost_lazy = lazy_splaynet(copy.deepcopy(network), sigma, alpha=100)
            print(f"Finished with total cost {total_cost_lazy}")
            print("Running Sliding Window SplayNet algorithm...")
            _, total_cost_sliding = sliding_window_splaynet(copy.deepcopy(network), sigma, slide=100)
            print(f"Finished with total cost {total_cost_sliding}")
            print(f"Finished running for {file}.\n\n")

            # Preparation for plotting
            combined_data.append((network_size, (total_cost_splaynet, total_cost_lazy, total_cost_sliding)))
            # Sort by size and separate
            combined_data.sort(key=lambda x: x[0])
            sizes = [item[0] for item in combined_data]
            costs = [item[1] for item in combined_data]
            # costs.append((total_cost_splaynet, total_cost_lazy, total_cost_sliding))

    if average:
        costs = [(cost[0] / size, cost[1] / size, cost[2] / size) for size, cost in zip(sizes, costs)]
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, [cost[0] for cost in costs], label='SplayNet')
    plt.plot(sizes, [cost[1] for cost in costs], label='Lazy SplayNet')
    plt.plot(sizes, [cost[2] for cost in costs], label='Sliding Window SplayNet')
    plt.xlabel('Size of Network')
    plt.ylabel('Total Cost of Operations')
    if log_scale:
        plt.title('Total Cost of Operations per Different Sizes of the Network with log scale')
        plt.yscale('log')
        plt.legend()
        plt.savefig("avg_output_logscale.png")
    else:
        plt.title('Total Cost of Operations per Different Sizes of the Network')
        plt.legend()
        plt.savefig("avg_output.png")


def main_temporal():
    # Infos for plotting
    log_scale = True  # True if results should be plotted with logarithmic scale
    average = True  # True if average cost should be plotted
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = "output"
    os.makedirs(output_folder, exist_ok=True)

    taus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    N = list(range(100, 1100, 100))
    combined_data = []

    for tau in taus:
        folder_name = "csv"
        os.makedirs(folder_name, exist_ok=True)

        for network_size in N:
            nodes = list(range(network_size))

            # Generate data
            df = generate_temporal_sequence(nodes, tau, length=10000)

            # Save DataFrame to CSV in the folder
            csv_filename = os.path.join(folder_name, f"data_{tau}_{network_size}.csv")
            df.to_csv(csv_filename, index=False)
            sigma, nodes = csv_to_sequence(csv_filename)

            # Initialize tree network
            network = SplayNetwork()
            network.insert_balanced_BST(nodes)

            print(f"Processing dataset for tau={tau} with {network_size} nodes.")

            # Apply algorithms, add loop for parameter finetuning here later
            print("Running SplayNet algorithm...")
            _, total_cost_splaynet = splaynet(copy.deepcopy(network), sigma)
            print(f"Finished with total cost {total_cost_splaynet}")

            print("Running Lazy SplayNet algorithm...")
            _, total_cost_lazy = lazy_splaynet(copy.deepcopy(network), sigma, alpha=100)
            print(f"Finished with total cost {total_cost_lazy}")

            print("Running Sliding Window SplayNet algorithm...")
            _, total_cost_sliding = sliding_window_splaynet(copy.deepcopy(network), sigma, slide=100)
            print(f"Finished with total cost {total_cost_sliding}")

            # Preparation for plotting
            combined_data.append((network_size, (total_cost_splaynet, total_cost_lazy, total_cost_sliding)))

    # Sort by size and separate
    combined_data.sort(key=lambda x: x[0])
    sizes = [item[0] for item in combined_data]
    costs = [item[1] for item in combined_data]

    if average:
        costs = [(cost[0] / size, cost[1] / size, cost[2] / size) for size, cost in zip(sizes, costs)]

    plt.figure(figsize=(10, 6))
    plt.plot(sizes, [cost[0] for cost in costs], label='SplayNet')
    plt.plot(sizes, [cost[1] for cost in costs], label='Lazy SplayNet')
    plt.plot(sizes, [cost[2] for cost in costs], label='Sliding Window SplayNet')
    plt.xlabel('Size of Network')
    plt.ylabel('Total Cost of Operations')

    if log_scale:
        plt.title('Total Cost of Operations per Different Sizes of the Network with log scale')
        plt.yscale('log')
        plt.legend()
        plot_filename = os.path.join(output_folder, f"avg_output_logscale_{timestamp}.png")
    else:
        plt.title('Total Cost of Operations per Different Sizes of the Network')
        plt.legend()
        plot_filename = os.path.join(output_folder, f"avg_output_{timestamp}.png")
    plt.savefig(plot_filename)

    for csv_file in os.listdir(folder_name):
        if csv_file.endswith(".csv"):
            df = pd.read_csv(csv_file)
            plt.figure(figsize=(10, 6))
            plt.scatter(df[0], df[1], alpha=0.6, marker='o')
            plt.xlabel('Source Node')
            plt.ylabel('Destination Node')
            plt.title(f'Scatter Plot for {os.path.basename(csv_file)}')
            output_name = os.path.join(output_folder, f"scatter_{os.path.basename(csv_file)[:-4]}_{timestamp}.png")
            plt.savefig(output_name)


if __name__ == '__main__':
    main_temporal()

    # print(sigma)
    # # online
    # network = sno()
    # network.insertBalancedBST(nodes)
    # print("Before sigma")
    # network.print_tree(network.root)
    # print("")
    # splaynet = splaynet.lsn_online(network, sigma, 5)
    # print("After sigma")
    # splaynet.print_tree(splaynet.root)

    # offline
    # t = SplayNetwork()
    # t.build_network(nodes)
    # t.print_tree(t.root)
    # splaynet = splaynet.lsn(t, sigma, 5)

    # filepath = "data/sample.csv"
    # # filepath = "data/Facebook.csv"
    # sigma, nodes = csv_to_sequence(filepath)
    # print(sigma)
    # # online
    # network = sno()
    # network.insertBalancedBST(nodes)
    # print("Before sigma")
    # network.print_tree(network.root)
    # print("")
    # splaynet = splaynet.lsn_online(network, sigma, 5)
    # print("After sigma")
    # splaynet.print_tree(splaynet.root)
