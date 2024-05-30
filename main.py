import copy
import os
import random
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from algorithms.algorithms import splaynet, lazy_splaynet, sliding_window_splaynet, variable_sliding_window_splaynet, \
    variable_sliding_window_splaynet_no_reset
from topology.CommunicationRequest import CommunicationRequest
from topology.SplayNetwork import SplayNetwork


# Helper functions
def plot_df_distribution(df):
    plt.figure(figsize=(10, 6))
    plt.scatter(df["src"], df["dst"], alpha=0.6, marker='o')
    plt.xlabel('Source Node')
    plt.ylabel('Destination Node')
    plt.title(f'Scatter Plot for distribution of df')
    # output_name = os.path.join(output_folder, f"scatter_{network_size}_{tau}_{timestamp}.png")
    # plt.savefig(output_name)
    # plt.close()
    plt.show()


def add_timestamp(name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return name + "_" + timestamp


def csv_to_sequence(path):
    data = pd.read_csv(path, usecols=["src", "dst"])
    communication_sequence = [CommunicationRequest(i, row[1], row[2]) for i, row in enumerate(data.itertuples(), 1)]
    all_nodes = pd.concat([data["src"], data["dst"]]).unique()
    # nodes = [Node(i) for i in all_nodes] # dieser Schritt erstellt alle Node Instanzen
    return communication_sequence, all_nodes


def generate_window_slides_list(window_size):
    percentages = [0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90]
    percentage_list = [int(window_size * p) for p in percentages]
    return percentage_list


# Request generators
def generate_sequence(name: str, nodes: [int], parameters: [int], length: int):
    if name == 'temporal':
        folder = f"{name}_data"
        os.makedirs(folder, exist_ok=True)

        for n in nodes:
            for param in parameters:
                temporal_requests = generate_temporal_sequence(n, param, length)
                filename = f"data_{param}_{n}.csv"
                pathname = os.path.join(folder, filename)
                temporal_requests.to_csv(pathname, index=False)

    elif name == 'spatial':
        folder = f"{name}_data"
        os.makedirs(folder, exist_ok=True)

        for n in nodes:
            for param in parameters:
                spatial_requests = generate_spatial_sequence(n, length, param)
                filename = f"data_{param}_{n}.csv"
                pathname = os.path.join(folder, filename)
                spatial_requests.to_csv(pathname, index=False)

    else:
        folder = None
        print("Invalid distribution name")
    return folder


def generate_temporal_sequence(nodes, tau, length):
    pairs = []
    prev_pair = None
    random.seed(42)
    for _ in range(length):
        if prev_pair is not None and random.random() < tau:
            u, v = prev_pair
        else:
            u, v = random.sample(list(range(1, nodes + 1)), 2)
            while u == v:
                v = random.choice(nodes)
        pairs.append((u, v))
        prev_pair = (u, v)
    df = pd.DataFrame(pairs, columns=["src", "dst"])
    # df["size"] = 1  # nicht gebraucht, aber damit dfs einheitlich sind
    return df


def generate_spatial_sequence(nodes, length, zipf_param):
    zipf_distribution = np.random.zipf(a=zipf_param, size=length)
    zipf_distribution = np.mod(zipf_distribution - 1, nodes) + 1

    src_nodes = np.random.randint(1, nodes + 1, size=length)
    dst_nodes = zipf_distribution

    for i in range(length):
        while dst_nodes[i] == src_nodes[i]:
            dst_nodes[i] = np.random.randint(1, nodes + 1)

    df = pd.DataFrame({'src': src_nodes, 'dst': dst_nodes})
    return df


# def generate_spatial_sequence(nodes, length, zipf_param):
#     zipf_distribution = np.random.zipf(a=zipf_param, size=length)
#
#     zipf_distribution = np.mod(zipf_distribution - 1, nodes) + 1
#
#     src_nodes = np.random.randint(1, nodes + 1, size=length)
#     dst_nodes = zipf_distribution
#
#     df = pd.DataFrame({'src': src_nodes, 'dst': dst_nodes})
#     return df


# Simulation runners
def run_simulation_splaynet(nodes, sigma):
    network = SplayNetwork()
    network.insert_balanced_BST(nodes)

    _, total_cost_splaynet, total_adjustment_cost_splaynet, total_routing_cost_splaynet = splaynet(network, sigma)
    splaynet_cost_results = [total_cost_splaynet, total_adjustment_cost_splaynet, total_routing_cost_splaynet]

    return splaynet_cost_results


def run_simulation_lazy(nodes, sigma, alpha_value):
    network = SplayNetwork()
    network.insert_balanced_BST(nodes)

    _, total_cost_lazy, total_adjustment_cost_lazy, total_routing_cost_lazy = lazy_splaynet(network, sigma,
                                                                                            alpha=alpha_value)
    lazy_cost_results = [total_cost_lazy, total_adjustment_cost_lazy, total_routing_cost_lazy]
    return lazy_cost_results


def run_simulation_sliding(nodes, sigma, window_size, slide_offset):
    network = SplayNetwork()
    network.insert_balanced_BST(nodes)

    _, total_cost_sliding, total_adjustment_cost_sliding, total_routing_cost_sliding = variable_sliding_window_splaynet(
        initial_topology=network, communication_sq=sigma,
        window_size=window_size, slide_offset=slide_offset)
    sliding_cost_results = [total_cost_sliding, total_adjustment_cost_sliding, total_routing_cost_sliding]
    return sliding_cost_results


def run_simulation_sliding_no_reset(nodes, sigma, window_size, slide_offset):
    network = SplayNetwork()
    network.insert_balanced_BST(nodes)

    _, total_cost_sliding, total_adjustment_cost_sliding, total_routing_cost_sliding = variable_sliding_window_splaynet_no_reset(
        initial_topology=network, communication_sq=sigma,
        window_size=window_size, slide_offset=slide_offset)
    sliding_cost_results = [total_cost_sliding, total_adjustment_cost_sliding, total_routing_cost_sliding]
    return sliding_cost_results


# depr
def run_simulation_temporal(nodes, tau, network_size, sigma, alpha, window_size):
    # Initialize tree network
    network = SplayNetwork()
    network.insert_balanced_BST(nodes)

    network_2 = copy.deepcopy(network)
    network_original = network

    print(f"Processing dataset for tau={tau} with {network_size} nodes.")

    # Apply algorithms, add loop for parameter finetuning here later
    print("Running SplayNet algorithm...")
    # _, total_cost_splaynet = splaynet(copy.deepcopy(network), sigma)
    _, total_cost_splaynet, total_adjustment_cost_splaynet, total_routing_cost_splaynet = splaynet(network_original,
                                                                                                   sigma)
    splaynet_cost_results = [total_cost_splaynet, total_adjustment_cost_splaynet, total_routing_cost_splaynet]
    network_original = network_2

    print(f"Finished with total cost {total_cost_splaynet}")

    print("Running Lazy SplayNet algorithm...")
    # _, total_cost_lazy = lazy_splaynet(copy.deepcopy(network), sigma, alpha=alpha)
    _, total_cost_lazy, total_adjustment_cost_lazy, total_routing_cost_lazy = lazy_splaynet(network_original, sigma,
                                                                                            alpha=alpha)
    lazy_cost_results = [total_cost_lazy, total_adjustment_cost_lazy, total_routing_cost_lazy]
    network_original = network_2

    print(f"Finished with total cost {total_cost_lazy}")

    # print("Running Sliding Window SplayNet algorithm...")
    # # _, total_cost_sliding = sliding_window_splaynet(copy.deepcopy(network), sigma, slide=slide)
    # _, total_cost_sliding = sliding_window_splaynet(network_original, sigma, slide=slide)
    # # network_original = network

    print("Running Sliding Window SplayNet algorithm...")
    # _, total_cost_sliding = sliding_window_splaynet(copy.deepcopy(network), sigma, slide=slide)
    _, total_cost_sliding, total_adjustment_cost_sliding, total_routing_cost_sliding = variable_sliding_window_splaynet(
        initial_topology=network_original, communication_sq=sigma,
        window_size=window_size, slide_offset=window_size // 2)
    sliding_cost_results = [total_cost_sliding, total_adjustment_cost_sliding, total_routing_cost_sliding]
    # network_original = network

    print(f"Finished with total cost {total_cost_sliding}")

    # Preparation for plotting
    # combined_data = [(network_size, (total_cost_splaynet, total_cost_lazy, total_cost_sliding))]
    # return total_cost_splaynet, total_cost_lazy, total_cost_sliding
    return splaynet_cost_results, lazy_cost_results, sliding_cost_results


# depr
# def run_simulation_temporal(nodes, tau, network_size, sigma, alpha, window_size):
#     # Initialize tree network
#     network = SplayNetwork()
#     network.insert_balanced_BST(nodes)
#
#     network_2 = copy.deepcopy(network)
#     network_original = network
#
#     print(f"Processing dataset for tau={tau} with {network_size} nodes.")
#
#     # Apply algorithms, add loop for parameter finetuning here later
#     print("Running SplayNet algorithm...")
#     # _, total_cost_splaynet = splaynet(copy.deepcopy(network), sigma)
#     _, total_cost_splaynet, total_adjustment_cost_splaynet, total_routing_cost_splaynet = splaynet(network_original,
#                                                                                                    sigma)
#     splaynet_cost_results = [total_cost_splaynet, total_adjustment_cost_splaynet, total_routing_cost_splaynet]
#     network_original = network_2
#
#     print(f"Finished with total cost {total_cost_splaynet}")
#
#     print("Running Lazy SplayNet algorithm...")
#     # _, total_cost_lazy = lazy_splaynet(copy.deepcopy(network), sigma, alpha=alpha)
#     _, total_cost_lazy, total_adjustment_cost_lazy, total_routing_cost_lazy = lazy_splaynet(network_original, sigma,
#                                                                                             alpha=alpha)
#     lazy_cost_results = [total_cost_lazy, total_adjustment_cost_lazy, total_routing_cost_lazy]
#     network_original = network_2
#
#     print(f"Finished with total cost {total_cost_lazy}")
#
#     # print("Running Sliding Window SplayNet algorithm...")
#     # # _, total_cost_sliding = sliding_window_splaynet(copy.deepcopy(network), sigma, slide=slide)
#     # _, total_cost_sliding = sliding_window_splaynet(network_original, sigma, slide=slide)
#     # # network_original = network
#
#     print("Running Sliding Window SplayNet algorithm...")
#     # _, total_cost_sliding = sliding_window_splaynet(copy.deepcopy(network), sigma, slide=slide)
#     _, total_cost_sliding, total_adjustment_cost_sliding, total_routing_cost_sliding = variable_sliding_window_splaynet(
#         initial_topology=network_original, communication_sq=sigma,
#         window_size=window_size, slide_offset=window_size // 2)
#     sliding_cost_results = [total_cost_sliding, total_adjustment_cost_sliding, total_routing_cost_sliding]
#     # network_original = network
#
#     print(f"Finished with total cost {total_cost_sliding}")
#
#     # Preparation for plotting
#     # combined_data = [(network_size, (total_cost_splaynet, total_cost_lazy, total_cost_sliding))]
#     # return total_cost_splaynet, total_cost_lazy, total_cost_sliding
#     return splaynet_cost_results, lazy_cost_results, sliding_cost_results

# depr
def run_simulation_spatial(nodes, zipf, network_size, sigma, alpha, window_size):
    # Initialize tree network
    network = SplayNetwork()
    network.insert_balanced_BST(nodes)

    network_original = network

    print(f"Processing dataset for zipf={zipf} with {network_size} nodes.")

    # Apply algorithms, add loop for parameter finetuning here later
    print("Running SplayNet algorithm...")
    # _, total_cost_splaynet = splaynet(copy.deepcopy(network), sigma)
    _, total_cost_splaynet = splaynet(network_original, sigma)
    network_original = network

    print(f"Finished with total cost {total_cost_splaynet}")

    print("Running Lazy SplayNet algorithm...")
    # _, total_cost_lazy = lazy_splaynet(copy.deepcopy(network), sigma, alpha=alpha)
    _, total_cost_lazy = lazy_splaynet(network_original, sigma, alpha=alpha)
    network_original = network

    print(f"Finished with total cost {total_cost_lazy}")

    # print("Running Sliding Window SplayNet algorithm...")
    # # _, total_cost_sliding = sliding_window_splaynet(copy.deepcopy(network), sigma, slide=slide)
    # _, total_cost_sliding = sliding_window_splaynet(network_original, sigma, slide=slide)
    # # network_original = network

    print("Running Sliding Window SplayNet algorithm...")
    # _, total_cost_sliding = sliding_window_splaynet(copy.deepcopy(network), sigma, slide=slide)
    _, total_cost_sliding = variable_sliding_window_splaynet(initial_topology=network_original, communication_sq=sigma,
                                                             window_size=window_size, slide_offset=window_size // 2)
    # network_original = network

    print(f"Finished with total cost {total_cost_sliding}")

    # Preparation for plotting
    # combined_data = [(network_size, (total_cost_splaynet, total_cost_lazy, total_cost_sliding))]
    return total_cost_splaynet, total_cost_lazy, total_cost_sliding


# Result calculating functions - Outputs are stored as csv for easier access for the plotting operations
def compute_temporal_results(tau_list, network_size_list,
                             data_folder, lazy_splaynet_alphas,
                             sliding_window_sizes,
                             sliding_offset_percentage_list):  # Ehemals: main_compute_temporal(log_scale_2, average_2)

    combinations = []
    columns = ["network_size", "tau", "algorithm", "adjustment_cost", "routing_cost", "total_cost", "alpha",
               "window_size", "sliding_offset"]
    results_df = pd.DataFrame(columns=columns)

    for tau in tau_list:
        for network_size in network_size_list:
            combinations.append(f"{tau}_{network_size}")

    # Ausgabe der Kombinationen
    for combination in combinations:
        tau = float(combination.split("_")[0])
        network_size = int(combination.split("_")[1])
        sigma, nodes = csv_to_sequence(f"{data_folder}/data_{combination}.csv")

        print(f"Processing dataset for tau={tau} with {network_size} nodes.")

        print("Running SplayNet algorithm...")
        results_splaynet = run_simulation_splaynet(nodes, sigma)
        # push to df
        results_df.loc[len(results_df)] = [network_size, tau, "splaynet", results_splaynet[1], results_splaynet[2],
                                           results_splaynet[0], None, None, None]

        for alpha_value in lazy_splaynet_alphas:
            print(f"Running Lazy SplayNet algorithm... with alpha={alpha_value}")
            results_lazy = run_simulation_lazy(nodes, sigma, alpha_value)
            # push to df
            results_df.loc[len(results_df)] = [network_size, tau, "lazy", results_lazy[1], results_lazy[2],
                                               results_lazy[0], alpha_value, None, None]

        for window_size in sliding_window_sizes:
            sliding_offset_list = [int(window_size * p) for p in sliding_offset_percentage_list]
            for slide_offset in sliding_offset_list:
                print(
                    f"Running SlidingWindow algorithm... with window size={window_size} and slide_offset={slide_offset}")
                results_sliding = run_simulation_sliding(nodes, sigma, window_size, slide_offset)
                results_df.loc[len(results_df)] = [network_size, tau, "sliding", results_sliding[1], results_sliding[2],
                                                   results_sliding[0], None, window_size, slide_offset]

                print(
                    f"Running SlidingWindow no reset algorithm... with window size={window_size} and slide_offset={slide_offset}")
                results_sliding_no_reset = run_simulation_sliding_no_reset(nodes, sigma, window_size, slide_offset)
                results_df.loc[len(results_df)] = [network_size, tau, "sliding_no_reset", results_sliding_no_reset[1],
                                                   results_sliding_no_reset[2],
                                                   results_sliding_no_reset[0], None, window_size, slide_offset]

    output_path = f"output/{add_timestamp("temporal_results")}.csv"
    results_df.to_csv(output_path)
    return output_path


def compute_spatial_results(zipf_list, network_size_list, data_folder, lazy_splaynet_alphas, sliding_window_sizes,
                            sliding_offset_percentage_list):
    combinations = []
    columns = ["network_size", "zipf", "algorithm", "adjustment_cost", "routing_cost", "total_cost", "alpha",
               "window_size", "sliding_offset"]
    results_df = pd.DataFrame(columns=columns)

    for zipf in zipf_list:
        for network_size in network_size_list:
            combinations.append(f"{zipf}_{network_size}")

    # Ausgabe der Kombinationen
    for combination in combinations:
        zipf = float(combination.split("_")[0])
        network_size = int(combination.split("_")[1])
        sigma, nodes = csv_to_sequence(f"{data_folder}/data_{combination}.csv")

        print(f"Processing dataset for zipf={zipf} with {network_size} nodes.")

        print("Running SplayNet algorithm...")
        results_splaynet = run_simulation_splaynet(nodes, sigma)
        # push to df
        results_df.loc[len(results_df)] = [network_size, zipf, "splaynet", results_splaynet[1], results_splaynet[2],
                                           results_splaynet[0], None, None, None]

        for alpha_value in lazy_splaynet_alphas:
            print(f"Running Lazy SplayNet algorithm... with alpha={alpha_value}")
            results_lazy = run_simulation_lazy(nodes, sigma, alpha_value)
            # push to df
            results_df.loc[len(results_df)] = [network_size, zipf, "lazy", results_lazy[1], results_lazy[2],
                                               results_lazy[0], alpha_value, None, None]

        for window_size in sliding_window_sizes:
            sliding_offset_list = [int(window_size * p) for p in sliding_offset_percentage_list]
            for slide_offset in sliding_offset_list:
                print(
                    f"Running SlidingWindow algorithm... with window size={window_size} and slide_offset={slide_offset}")
                results_sliding = run_simulation_sliding(nodes, sigma, window_size, slide_offset)
                results_df.loc[len(results_df)] = [network_size, zipf, "sliding", results_sliding[1],
                                                   results_sliding[2],
                                                   results_sliding[0], None, window_size, slide_offset]

                print(
                    f"Running SlidingWindow no reset algorithm... with window size={window_size} and slide_offset={slide_offset}")
                results_sliding_no_reset = run_simulation_sliding_no_reset(nodes, sigma, window_size, slide_offset)
                results_df.loc[len(results_df)] = [network_size, zipf, "sliding_no_reset", results_sliding_no_reset[1],
                                                   results_sliding_no_reset[2],
                                                   results_sliding_no_reset[0], None, window_size, slide_offset]

    output_path = f"output/{add_timestamp("spatial_results")}.csv"
    results_df.to_csv(output_path)
    return output_path


# depr
# def compute_spatial_results(zipf_params, network_size_list, data_folder):  # Ehemals: main(log2, average2)
#     # anhand parameter daten einlesen
#     # zipf_params = ["1.1", "1.25", "1.5", "1.75", "2.0", "2.25", "2.5"]
#     # network_size_list = ["100", "200", "300", "400", "500", "600", "700", "800", "900", "1000"]
#     # tau_list = ["0.1"]
#     # network_size_list = ["100"]
#
#     combinations = []
#     results_df = pd.DataFrame(
#         columns=["network_size", "zipf", "total_cost_splaynet", "total_cost_lazy", "total_cost_sliding"])
#
#     for zipf in zipf_params:
#         for network_size in network_size_list:
#             combinations.append(f"{zipf}_{network_size}")
#
#     # All variable combinations
#     for combination in combinations:
#         zipf = float(combination.split("_")[0])
#         network_size = int(combination.split("_")[1])
#         # sigma, nodes = csv_to_sequence(f"csv_spatial/data_{combination}.csv")
#         sigma, nodes = csv_to_sequence(f"{data_folder}/data_{combination}.csv")
#         total_cost_splaynet, total_cost_lazy, total_cost_sliding = run_simulation_spatial(nodes, zipf, network_size,
#                                                                                           sigma, 100,
#                                                                                           100)
#         results_df.loc[len(results_df)] = [network_size, zipf, total_cost_splaynet, total_cost_lazy, total_cost_sliding]
#
#     output_path = f"output/{add_timestamp("spatial_results")}.csv"
#     results_df.to_csv(output_path)
#     return output_path


# depr
def compute_temporal_results_old(tau_list, network_size_list,
                                 data_folder, lazy_splaynet_alpha,
                                 sliding_window_size):  # Ehemals: main_compute_temporal(log_scale_2, average_2)
    # Für temporal
    # anhand parameter daten einlesen
    # tau_list = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.5", "0.6", "0.7", "0.8", "0.9"]
    # network_size_list = ["100", "200", "300", "400", "500", "600", "700", "800", "900", "1000"]
    # tau_list = ["0.1"]
    # network_size_list = ["100"]

    combinations = []
    results_df = pd.DataFrame(
        columns=["network_size", "tau", "adjustment_cost_splaynet", "routing_cost_splaynet", "total_cost_splaynet",
                 "adjustment_cost_lazy", "routing_cost_lazy", "total_cost_lazy", "adjustment_cost_sliding",
                 "routing_cost_sliding", "total_cost_sliding"])

    # columns2= ["network_size", "tau", "algorithm", "adjustment_cost", "routing_cost", "total_cost", "alpha", "window_size", "sliding_offset"]

    # "adjustment_cost_splaynet", "routing_cost_splaynet", "total_cost_splaynet"
    # "adjustment_cost_splaynet<a-value>", "routing_cost_splaynet<a-value>", "total_cost_splaynet<a-value>"
    # "adjustment_cost_splaynet<window_size>_<sliding_offset>", "routing_cost_sliding<window_size>_<sliding_offset>", "total_cost_sliding<window_size>_<sliding_offset>"

    # network_size, tau, algorithm, adjustment_cost, routing_cost, total_cost, alpha, window-size, sliding_offset

    # New DF including results of new ALG
    # results_df = pd.DataFrame(
    #     columns=["network_size", "tau", "adjustment_cost_splaynet", "routing_cost_splaynet", "total_cost_splaynet",
    #              "adjustment_cost_lazy", "routing_cost_lazy", "total_cost_lazy", "adjustment_cost_sliding_reset",
    #              "routing_cost_sliding_reset", "total_cost_sliding_reset", "adjustment_cost_sliding_no_reset",
    #              "routing_cost_sliding_no_reset", "total_cost_sliding_no_reset"])

    for tau in tau_list:
        for network_size in network_size_list:
            combinations.append(f"{tau}_{network_size}")

    # Ausgabe der Kombinationen
    for combination in combinations:
        tau = float(combination.split("_")[0])
        network_size = int(combination.split("_")[1])
        sigma, nodes = csv_to_sequence(f"{data_folder}/data_{combination}.csv")
        # total_cost_splaynet, total_cost_lazy, total_cost_sliding = run_simulation_temporal(nodes, tau, network_size,
        #                                                                                    sigma, 100,
        #                                                                                    100)
        results_splaynet, results_lazy, results_sliding = run_simulation_temporal(nodes, tau, network_size,
                                                                                  sigma, lazy_splaynet_alpha,
                                                                                  sliding_window_size)
        # results_df.loc[len(results_df)] = [network_size, tau, total_cost_splaynet, total_cost_lazy,
        # total_cost_sliding]
        results_df.loc[len(results_df)] = [network_size, tau, results_splaynet[2], results_splaynet[1],
                                           results_splaynet[0], results_lazy[2], results_lazy[1], results_lazy[0],
                                           results_sliding[2], results_sliding[1], results_sliding[0]]

    output_path = f"output/{add_timestamp("temporal_results")}.csv"
    results_df.to_csv(output_path)
    return output_path

    # print(sigma, nodes)

    # logscale total cost
    # logscale average cost
    # total cost normal scale
    # average cost total cost
    # Spatial locality (zip distribution) (die Graphen auf dem Papier )

    # - [ ] Neuer graph mit Divide by the number of requests statt number of nodes of network


# Plotting functions
def plot_x_network_size_y_total_cost(df, log_scale_bool):
    plt.figure(figsize=(10, 6))

    # Filter data for each algorithm
    df_splaynet = df[df['algorithm'] == 'splaynet']
    df_lazy = df[df['algorithm'] == 'lazy']
    df_sliding = df[df['algorithm'] == 'sliding']
    df_sliding_no_reset = df[df['algorithm'] == 'sliding_no_reset']

    # Plot for total_cost_splaynet
    plt.plot(df_splaynet['network_size'], df_splaynet['total_cost'], label='SplayNet', marker='o')

    # Plot for total_cost_lazy
    plt.plot(df_lazy['network_size'], df_lazy['total_cost'], label='Lazy', marker='s')

    # Plot for total_cost_sliding
    plt.plot(df_sliding['network_size'], df_sliding['total_cost'], label='Sliding', marker='^')

    # Plot for total_cost_sliding
    plt.plot(df_sliding_no_reset['network_size'], df_sliding_no_reset['total_cost'], label='Sliding no reset',
             marker='x')

    # Axis labels and title
    plt.xlabel('Network Size')
    plt.ylabel('Total Cost')
    plt.title('Total Cost vs Network Size')

    # Show legend
    plt.legend()

    # Show grid
    plt.grid(True)

    # Log scale if required
    if log_scale_bool:
        plt.yscale('log')

    # Save plot with a timestamp in the filename
    type_of_data = "spatial" if "zipf" in df.columns else "temporal"
    filename = f"output/{type_of_data}_{add_timestamp('x_network_size_y_total_cost_log_scale')}.png" if log_scale_bool else f"output/{type_of_data}_{add_timestamp('x_network_size_y_total_cost')}.png"
    plt.savefig(filename)
    plt.show()


def plot_x_network_size_y_average_cost(df, log_scale_bool):
    avg_df = df.groupby(['network_size', 'algorithm']).mean().reset_index()

    # Plot erstellen
    plt.figure(figsize=(10, 6))

    # Filter data for each algorithm
    df_splaynet = avg_df[avg_df['algorithm'] == 'splaynet']
    df_lazy = avg_df[avg_df['algorithm'] == 'lazy']
    df_sliding = avg_df[avg_df['algorithm'] == 'sliding']
    df_sliding_no_reset = avg_df[avg_df['algorithm'] == 'sliding_no_reset']

    # Plot for total_cost_splaynet
    plt.plot(df_splaynet['network_size'], df_splaynet['total_cost'], label='SplayNet', marker='o')

    # Plot for total_cost_lazy
    plt.plot(df_lazy['network_size'], df_lazy['total_cost'], label='Lazy', marker='s')

    # Plot for total_cost_sliding
    plt.plot(df_sliding['network_size'], df_sliding['total_cost'], label='Sliding', marker='^')

    plt.plot(df_sliding_no_reset['network_size'], df_sliding_no_reset['total_cost'], label='Sliding no reset',
             marker='x')

    # Axis labels and title
    plt.xlabel('Network Size')
    plt.ylabel('Average Total Cost')
    plt.title('Average Total Cost vs Network Size')

    # Show legend
    plt.legend()

    # Show grid
    plt.grid(True)

    # Log scale if required
    if log_scale_bool:
        plt.yscale('log')

    # Save plot with a timestamp in the filename
    type_of_data = "spatial" if "zipf" in df.columns else "temporal"
    filename = f"output/{type_of_data}_{add_timestamp('x_network_size_y_average_cost_log_scale')}.png" if log_scale_bool else f"output/{type_of_data}_{add_timestamp('x_network_size_y_average_cost')}.png"
    plt.savefig(filename)
    plt.show()


def plot_x_zipf_or_tau_y_total_cost(df, network_size):
    zipf_or_tau = "zipf" if "zipf" in df.columns else "tau"
    filtered_df = df[df['network_size'] == network_size]

    if filtered_df.empty:
        print(f"No data available for network size {network_size}")
        return

    # Plot erstellen
    plt.figure(figsize=(10, 6))

    # Filter data for each algorithm
    df_splaynet = filtered_df[filtered_df['algorithm'] == 'splaynet']
    df_lazy = filtered_df[filtered_df['algorithm'] == 'lazy']
    df_sliding = filtered_df[filtered_df['algorithm'] == 'sliding']
    df_sliding_no_reset = filtered_df[filtered_df['algorithm'] == 'sliding_no_reset']

    # Plot for total_cost_splaynet
    plt.plot(df_splaynet[zipf_or_tau], df_splaynet['total_cost'], label='SplayNet', marker='o')

    # Plot for total_cost_lazy
    plt.plot(df_lazy[zipf_or_tau], df_lazy['total_cost'], label='Lazy', marker='s')

    # Plot for total_cost_sliding
    plt.plot(df_sliding[zipf_or_tau], df_sliding['total_cost'], label='Sliding', marker='^')

    # Plot for total_cost_sliding_no_reset
    plt.plot(df_sliding_no_reset[zipf_or_tau], df_sliding_no_reset['total_cost'], label='Sliding No Reset', marker='x')

    # Axis labels and title
    plt.xlabel(zipf_or_tau)
    plt.ylabel('Total Cost')
    plt.title(f'Total Cost vs {zipf_or_tau} for Network Size {network_size}')

    # Show legend
    plt.legend()

    # Show grid
    plt.grid(True)
    type_text = "temporal" if zipf_or_tau == "tau" else "spatial"

    # Save plot with a timestamp in the filename
    plt.savefig(f"output/{type_text}_x_{zipf_or_tau}_y_total_cost_network_size_{network_size}{add_timestamp('')}.png")
    # Show plot
    plt.show()


def plot_x_zipf_or_tau_y_average_cost(df, network_size):
    zipf_or_tau = "zipf" if "zipf" in df.columns else "tau"
    filtered_df = df[df['network_size'] == network_size]

    if filtered_df.empty:
        print(f"No data available for network size {network_size}")
        return

    avg_df = filtered_df.groupby([zipf_or_tau, 'algorithm']).mean().reset_index()

    # Plot erstellen
    plt.figure(figsize=(10, 6))

    # Filter data for each algorithm
    df_splaynet = avg_df[avg_df['algorithm'] == 'splaynet']
    df_lazy = avg_df[avg_df['algorithm'] == 'lazy']
    df_sliding = avg_df[avg_df['algorithm'] == 'sliding']
    df_sliding_no_reset = avg_df[avg_df['algorithm'] == 'sliding_no_reset']

    # Plot for total_cost_splaynet
    plt.plot(df_splaynet[zipf_or_tau], df_splaynet['total_cost'], label='SplayNet', marker='o')

    # Plot for total_cost_lazy
    plt.plot(df_lazy[zipf_or_tau], df_lazy['total_cost'], label='Lazy', marker='s')

    # Plot for total_cost_sliding
    plt.plot(df_sliding[zipf_or_tau], df_sliding['total_cost'], label='Sliding', marker='^')

    # Plot for total_cost_sliding_no_reset
    plt.plot(df_sliding_no_reset[zipf_or_tau], df_sliding_no_reset['total_cost'], label='Sliding No Reset', marker='x')

    # Axis labels and title
    plt.xlabel(zipf_or_tau.capitalize())
    plt.ylabel('Average Total Cost')
    plt.title(f'Average Total Cost vs {zipf_or_tau.capitalize()} for Network Size {network_size}')

    # Show legend
    plt.legend()

    # Show grid
    plt.grid(True)
    type_text = "temporal" if zipf_or_tau == "tau" else "spatial"

    # Save plot with a timestamp in the filename
    plt.savefig(f"output/{type_text}_x_{zipf_or_tau}_y_average_cost_network_size_{network_size}{add_timestamp('')}.png")
    # Show plot
    plt.show()


def plot_avg_cost_comparison(df, algorithms):
    zipf_or_tau = "zipf" if "zipf" in df.columns else "tau"

    # Durchschnittliche Werte für jede Network Size und Algorithmus berechnen
    avg_df = df.groupby(['network_size', 'algorithm']).mean().reset_index()

    # Plot erstellen
    plt.figure(figsize=(10, 6))

    # Plot für jeden Algorithmus in der Liste
    for algorithm in algorithms:
        filtered_df = avg_df[avg_df['algorithm'] == algorithm]
        plt.plot(filtered_df['network_size'], filtered_df['total_cost'], label=algorithm.capitalize(), marker='o')

    # Achsenbeschriftungen und Titel
    plt.xlabel('Network Size')
    plt.ylabel('Average Total Cost')
    plt.title('Average Total Cost Comparison')

    # Legende anzeigen
    plt.legend()

    # Grid anzeigen
    plt.grid(True)

    type_text = "temporal" if zipf_or_tau == "tau" else "spatial"
    plt.savefig(
        f"output/{type_text}_x_network_size_y_average_cost_with_algorithms_{'_'.join(algorithms)}_{add_timestamp('')}.png")

    # Plot anzeigen
    plt.show()


def plot_cost_ratio(df, network_size, cost_type1, cost_type2, algorithms):
    # Filtere das DataFrame für die angegebene network_size
    filtered_df = df[df['network_size'] == network_size]

    if filtered_df.empty:
        print(f"No data available for network size {network_size}")
        return

    # Plot erstellen
    plt.figure(figsize=(10, 6))

    # Plot für das Kostenverhältnis für jeden Algorithmus in der Liste
    for algorithm in algorithms:
        algorithm_df = filtered_df[filtered_df['algorithm'] == algorithm]

        if not algorithm_df.empty:
            # Berechne das Kostenverhältnis
            cost_ratio = algorithm_df[cost_type1] / algorithm_df[cost_type2]

            # Plot für das Kostenverhältnis
            plt.plot(algorithm_df['tau'], cost_ratio, label=f'{algorithm} {cost_type1}/{cost_type2}', marker='o')

    # Achsenbeschriftungen und Titel
    plt.xlabel('Tau')
    plt.ylabel(f'{cost_type1}/{cost_type2}')
    plt.title(f'Cost Ratio ({cost_type1}/{cost_type2}) vs Tau for Network Size {network_size}')

    # Logarithmische Skala auf der y-Achse setzen
    plt.yscale('log')

    # Grid anzeigen
    plt.grid(True)

    # Plot anzeigen
    plt.legend()
    plt.show()


def test_main():
    filepath = "data/sample.csv"
    sigma, nodes = csv_to_sequence(filepath)
    network = SplayNetwork()
    network.insert_balanced_BST(nodes)
    network.print_tree(network.root)

    print(f"Processing dataset: {filepath} with {len(nodes)} nodes.")
    # Apply algorithms, add loop for parameter finetuning here later

    print("Running SplayNet algorithm...")
    print("Netwerk am Anfang")
    network.print_tree(network.root)
    _, total_cost_splaynet = splaynet(copy.deepcopy(network), sigma)
    print(f"Finished with total cost {total_cost_splaynet}")

    print("Running Lazy SplayNet algorithm...")
    print("Netwerk am Anfang")
    network.print_tree(network.root)
    _, total_cost_lazy = lazy_splaynet(copy.deepcopy(network), sigma, alpha=100)
    print(f"Finished with total cost {total_cost_lazy}")

    print("Running Sliding Window SplayNet algorithm...")
    print("Netwerk am Anfang")
    network.print_tree(network.root)
    _, total_cost_sliding = sliding_window_splaynet(copy.deepcopy(network), sigma, slide=100)
    print(f"Finished with total cost {total_cost_sliding}")
    print(f"Finished running for {filepath}.\n\n")

    print("Running Variable Sliding Window SplayNet algorithm...")
    print("Netwerk am Anfang")
    network.print_tree(network.root)
    _, total_cost_sliding = variable_sliding_window_splaynet(copy.deepcopy(network), sigma, window_size=100,
                                                             slide_offset=50)
    print(f"Finished with total cost {total_cost_sliding}")
    print(f"Finished running for {filepath}.\n\n")


def spatial_sequence_wrapper(length):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = "output"
    os.makedirs(output_folder, exist_ok=True)

    zipf_params = [1.1, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]
    N = list(range(100, 1100, 100))
    combined_data = []

    for zipf_param in zipf_params:
        folder_name = "csv_spatial"
        os.makedirs(folder_name, exist_ok=True)

        for network_size in N:
            nodes = network_size

            # Generate data
            df = generate_spatial_sequence(nodes, length, zipf_param)

            # Save DataFrame to CSV in the folder
            csv_filename = os.path.join(folder_name, f"data_{zipf_param}_{network_size}.csv")
            df.to_csv(csv_filename, index=False)


# Main functions
def real_data_main(log_scale, average):
    # Infos for plotting
    # True if average cost should be plotted

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


if __name__ == '__main__':
    # Algorithm specific variables
    # threshold_alpha = 100  # Lazy SplayNet
    threshold_alpha_list = [100]
    # sliding_window_size = 100  # Sliding Window
    sliding_window_size_list = [100]
    sliding_offset_percentages = [0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90]

    # Bool values and variables, that determine which parts of the workflow are executed
    # Generator booleans
    generate_spatial = False  # Set variables to True, if the datasets should be generated
    generate_temporal = False

    # Compute simulation booleans
    run_simulation = False

    # Plot graphs boolean
    plot_temporal = True
    plot_spatial = False

    # Plotting variables
    log_scale = True  # True if results should be plotted with logarithmic scale
    average = True  # True if average cost should be plotted
    slice_temporal = ([None], [None])  # Variable to slice result as wished: ([network_size_slices], [parameter_slices])
    slice_spatial = ([None], [None])

    # Parameters - can be adjusted as desired
    zipf_parameters = [1.1, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]
    tau_parameters = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    sequence_length = 100000

    # node_ranges = (100, 1100, 100)  # can either a single value [size] or as a tuple (start, stop, step)
    node_ranges = [1000]

    if len(node_ranges) == 1:
        size_list = [node_ranges[0]]
    elif 1 < len(node_ranges) < 4:
        size_list = list(range(*node_ranges))
    else:
        print("Invalid node range")
        size_list = []

    # Generate communication sequence datasets
    if generate_spatial:
        print("Generating communication sequences for different spatial localities ...")
        spatial_sequences_folder = generate_sequence('spatial', size_list, zipf_parameters, sequence_length)
        print(f"Completed. Result(s) can be found in the folder {spatial_sequences_folder}. \n")
    else:
        spatial_sequences_folder = 'spatial_data'
        print(f"Communication requests are taken from the folder {spatial_sequences_folder}.\n")

    if generate_temporal:
        print("Generating temporal sequences for different temporal localities ...")
        temporal_sequences_folder = generate_sequence('temporal', size_list, tau_parameters, sequence_length)
        print(f"Completed. Result(s) can be found in the folder {temporal_sequences_folder}.\n")
    else:
        temporal_sequences_folder = 'temporal_data'
        print(f"Communication requests are taken from the folder '{temporal_sequences_folder}'\n.")

    # Run simulations on datasets - results are saved in a csv (see path) and can be used multiple times
    temporal_output_path = "output/temporal_results_20240529_210645.csv"
    spatial_output_path = "output/spatial_results_20240529_231840.csv"
    if run_simulation:
        '''test_main()'''  # runs simulation of sample.csv
        '''real_data_main(log_scale, average)'''  # runs simulations on all real traces in /csv
        # temporal_output_path = compute_temporal_results(tau_parameters, size_list, temporal_sequences_folder,
        #                                                 lazy_splaynet_alphas=threshold_alpha_list,
        #                                                 sliding_window_sizes=sliding_window_size_list,
        #                                                 sliding_offset_percentage_list=sliding_offset_percentages)

        spatial_output_path = compute_spatial_results(zipf_parameters, size_list, spatial_sequences_folder,
                                                      lazy_splaynet_alphas=threshold_alpha_list,
                                                      sliding_window_sizes=sliding_window_size_list,
                                                      sliding_offset_percentage_list=sliding_offset_percentages)
        # pass
    # Add path of desired simulation

    temporal_df = pd.read_csv(temporal_output_path)
    spatial_df = pd.read_csv(spatial_output_path)

    # Evaluate and generate plots
    # Evaluate performance of communication sequences of different localities
    plot_x_network_size_y_total_cost(temporal_df, log_scale)
    plot_x_network_size_y_total_cost(temporal_df, not log_scale)
    if average:
        plot_x_network_size_y_average_cost(temporal_df, log_scale)
        plot_x_network_size_y_average_cost(temporal_df, not log_scale)

    plot_x_zipf_or_tau_y_total_cost(temporal_df, 1000)
    # plot_x_zipf_or_tau_y_total_cost(temporal_df, 200)
    # plot_x_zipf_or_tau_y_total_cost(temporal_df, 300)
    # plot_x_zipf_or_tau_y_total_cost(temporal_df, 400)

    plot_x_zipf_or_tau_y_average_cost(temporal_df, 1000)

    plot_avg_cost_comparison(temporal_df, ["lazy", "sliding", "sliding_no_reset"])
    plot_avg_cost_comparison(temporal_df, ["sliding", "sliding_no_reset"])

    # plot_cost_ratio(temporal_df, 100, "total_cost_lazy", "total_cost_splaynet")

    # plot_cost_ratio(temporal_df, 1000, "total_cost", "routing_cost", ["lazy", "sliding", "sliding_no_reset"])

    # Spatial
    plot_x_network_size_y_total_cost(spatial_df, log_scale)
    plot_x_network_size_y_total_cost(spatial_df, not log_scale)

    if average:
        plot_x_network_size_y_average_cost(spatial_df, log_scale)
        plot_x_network_size_y_average_cost(spatial_df, not log_scale)

    plot_x_zipf_or_tau_y_total_cost(spatial_df, 1000)
    plot_x_zipf_or_tau_y_average_cost(spatial_df, 1000)

    plot_avg_cost_comparison(spatial_df, ["lazy", "sliding", "sliding_no_reset"])
    plot_avg_cost_comparison(spatial_df, ["sliding", "sliding_no_reset"])

    # main_temporal(log_scale, average)
    # main(log_scale, average)

    # spatial_sequence_wrapper(10000, average)
    # main_compute_spatial(log_scale, average)

    # mit option 2
    # temporal_df = pd.read_csv("output/results_20240521_182655.csv")
    # plot_x_network_size_y_total_cost(temporal_df, log_scale)
    # plot_x_network_size_y_average_cost(temporal_df, log_scale)
    # plot_x_network_size_y_total_cost(temporal_df, not log_scale)
    # plot_x_network_size_y_average_cost(temporal_df, not log_scale)
    # plot_x_tau_y_total_cost(temporal_df, 100)
    # plot_x_tau_y_total_cost(temporal_df, 200)
    # plot_x_tau_y_total_cost(temporal_df, 300)
    # plot_x_tau_y_total_cost(temporal_df, 400)

    # plot_avg_cost_comparison(temporal_df, True)
    # plot_avg_cost_comparison(temporal_df, False)

    # plot_cost_ratio(temporal_df, 100, "total_cost_lazy", "total_cost_splaynet")

    # Alt
    # temporal_df = pd.read_csv("output/results.csv")
    # plot_x_network_size_y_total_cost(temporal_df, log_scale)
    # plot_x_network_size_y_average_cost(temporal_df, log_scale)
    # plot_x_network_size_y_total_cost(temporal_df, not log_scale)
    # plot_x_network_size_y_average_cost(temporal_df, not log_scale)
    # plot_x_tau_y_total_cost(temporal_df, 100)
    # plot_x_tau_y_total_cost(temporal_df, 200)
    # plot_x_tau_y_total_cost(temporal_df, 300)
    # plot_x_tau_y_total_cost(temporal_df, 400)

    # plot_avg_cost_comparison(temporal_df, True)
    # plot_avg_cost_comparison(temporal_df, False)

    # plot_cost_ratio(temporal_df, 100, "total_cost_lazy", "total_cost_splaynet")

    # def main_temporal(log_scale, average):
    #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #     output_folder = "output"
    #     os.makedirs(output_folder, exist_ok=True)
    #
    #     taus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    #     N = list(range(100, 1100, 100))
    #     combined_data = []
    #
    #     for tau in taus:
    #         folder_name = "csv"
    #         os.makedirs(folder_name, exist_ok=True)
    #
    #         for network_size in N:
    #             nodes = list(range(network_size))
    #
    #             # Generate data
    #             df = generate_temporal_sequence(nodes, tau, length=1000000)
    #
    #             # Save DataFrame to CSV in the folder
    #             csv_filename = os.path.join(folder_name, f"data_{tau}_{network_size}.csv")
    #             df.to_csv(csv_filename, index=False)
    #             sigma, nodes = csv_to_sequence(csv_filename)
    #
    #             # Initialize tree network
    #             network = SplayNetwork()
    #             network.insert_balanced_BST(nodes)
    #
    #             print(f"Processing dataset for tau={tau} with {network_size} nodes.")
    #
    #             # Apply algorithms, add loop for parameter finetuning here later
    #             print("Running SplayNet algorithm...")
    #             _, total_cost_splaynet = splaynet(copy.deepcopy(network), sigma)
    #             print(f"Finished with total cost {total_cost_splaynet}")
    #
    #             print("Running Lazy SplayNet algorithm...")
    #             _, total_cost_lazy = lazy_splaynet(copy.deepcopy(network), sigma, alpha=100)
    #             print(f"Finished with total cost {total_cost_lazy}")
    #
    #             print("Running Sliding Window SplayNet algorithm...")
    #             _, total_cost_sliding = sliding_window_splaynet(copy.deepcopy(network), sigma, slide=100)
    #             print(f"Finished with total cost {total_cost_sliding}")
    #
    #             # Preparation for plotting
    #             combined_data.append((network_size, (total_cost_splaynet, total_cost_lazy, total_cost_sliding)))
    #
    #     # Sort by size and separate
    #     combined_data.sort(key=lambda x: x[0])
    #     # sizes = [item[0] for item in combined_data]
    #     sizes = [10] * len(combined_data)
    #     costs = [item[1] for item in combined_data]
    #
    #     if average:
    #         costs = [(cost[0] / size, cost[1] / size, cost[2] / size) for size, cost in zip(sizes, costs)]
    #
    #     plt.figure(figsize=(10, 6))
    #     plt.plot(sizes, [cost[0] for cost in costs], label='SplayNet')
    #     plt.plot(sizes, [cost[1] for cost in costs], label='Lazy SplayNet')
    #     plt.plot(sizes, [cost[2] for cost in costs], label='Sliding Window SplayNet')
    #     plt.xlabel('Size of Network')
    #     plt.ylabel('Total Cost of Operations')
    #
    #     if log_scale:
    #         plt.title('Total Cost of Operations per Different Sizes of the Network with log scale')
    #         plt.yscale('log')
    #         plt.legend()
    #         plot_filename = os.path.join(output_folder, f"avg_output_logscale_{timestamp}.png")
    #     else:
    #         plt.title('Total Cost of Operations per Different Sizes of the Network')
    #         plt.legend()
    #         plot_filename = os.path.join(output_folder, f"avg_output_{timestamp}.png")
    #     plt.savefig(plot_filename)
    #
    #     for csv_file in os.listdir(folder_name):
    #         if csv_file.endswith(".csv"):
    #             df = pd.read_csv(csv_file)
    #             plt.figure(figsize=(10, 6))
    #             plt.scatter(df[0], df[1], alpha=0.6, marker='o')
    #             plt.xlabel('Source Node')
    #             plt.ylabel('Destination Node')
    #             plt.title(f'Scatter Plot for {os.path.basename(csv_file)}')
    #             output_name = os.path.join(output_folder, f"scatter_{os.path.basename(csv_file)[:-4]}_{timestamp}.png")
    #             plt.savefig(output_name)
