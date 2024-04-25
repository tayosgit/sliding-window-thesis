from algorithms.algorithms import splaynet, lazy_splaynet, sliding_window_splaynet
from topology.CommunicationRequest import CommunicationRequest
from topology.SplayNetwork import SplayNetwork
import matplotlib.pyplot as plt
import pandas as pd
import copy
import os


def csv_to_sequence(path):
    data = pd.read_csv(path, usecols=["src", "dst"])
    communication_sequence = [CommunicationRequest(i, row[1], row[2]) for i, row in enumerate(data.itertuples(), 1)]
    all_nodes = pd.concat([data["src"], data["dst"]]).unique()
    # nodes = [Node(i) for i in all_nodes] # dieser Schritt erstellt alle Node Instanzen
    return communication_sequence, all_nodes


if __name__ == '__main__':
    combined_data = []
    # costs = []
    # sizes = []

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
            # Sort by size and seperate
            combined_data.sort(key=lambda x: x[0])
            sizes = [item[0] for item in combined_data]
            costs = [item[1] for item in combined_data]
            # costs.append((total_cost_splaynet, total_cost_lazy, total_cost_sliding))
            # sizes.append(network_size)

    # # Plotting results
    # plt.figure(figsize=(10, 6))
    # plt.plot(sizes, [cost[0] for cost in costs], label='SplayNet')
    # plt.plot(sizes, [cost[1] for cost in costs], label='Lazy SplayNet')
    # plt.plot(sizes, [cost[2] for cost in costs], label='Sliding Window SplayNet')
    # plt.xlabel('Size of Network')
    # plt.ylabel('Total Cost of Operations')
    # plt.title('Total Cost of Operations per Different Sizes of the Network')
    # plt.legend()
    # plt.savefig("output.png")

    plt.figure(figsize=(10, 6))
    plt.plot(sizes, [cost[0] for cost in costs], label='SplayNet')
    plt.plot(sizes, [cost[1] for cost in costs], label='Lazy SplayNet')
    plt.plot(sizes, [cost[2] for cost in costs], label='Sliding Window SplayNet')
    plt.xlabel('Size of Network')
    plt.ylabel('Total Cost of Operations')
    plt.title('Total Cost of Operations per Different Sizes of the Network')
    plt.yscale('log')  # Setzt die y-Achse auf eine logarithmische Skala
    plt.legend()
    plt.savefig("output_logscale.png")


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
