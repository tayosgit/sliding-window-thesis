import copy
import os
import random
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
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
    return communication_sequence, all_nodes


def csv_to_frequency_sequence(path):
    data = pd.read_csv(path, usecols=["src", "dst"])

    # Count frequencies of each node
    freq_src = data['src'].value_counts()
    freq_dst = data['dst'].value_counts()

    # Sum frequencies
    total_freq = freq_src.add(freq_dst, fill_value=0)

    # Sort nodes by frequency in descending order
    sorted_nodes = total_freq.sort_values(ascending=False).index.tolist()

    communication_sequence = [CommunicationRequest(i, row[1], row[2]) for i, row in enumerate(data.itertuples(), 1)]
    return communication_sequence, sorted_nodes


def csv_to_shuffled_sequence(path):
    data = pd.read_csv(path, usecols=["src", "dst"])
    communication_sequence = [CommunicationRequest(i, row[1], row[2]) for i, row in enumerate(data.itertuples(), 1)]
    all_nodes = pd.concat([data["src"], data["dst"]]).unique()
    random.shuffle(all_nodes)
    return communication_sequence, all_nodes


def csv_to_optimal_sequence(path):
    data = pd.read_csv(path, usecols=["src", "dst"])
    communication_sequence = [CommunicationRequest(i, row[1], row[2]) for i, row in enumerate(data.itertuples(), 1)]
    all_nodes = pd.concat([data["src"], data["dst"]]).unique()
    node_index = {node: i for i, node in enumerate(sorted(all_nodes))}
    node_count = len(all_nodes)
    node_weights = [[0] * node_count for _ in range(node_count)]

    for request in communication_sequence:
        src_idx = node_index[request.src]
        dst_idx = node_index[request.dst]
        if src_idx < dst_idx:
            node_weights[src_idx][dst_idx] += 1
        else:
            node_weights[dst_idx][src_idx] += 1

    return communication_sequence, all_nodes, node_weights


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


# Simulation runners
def run_simulation_splaynet(nodes, sigma, tree_type, node_weights):
    network = SplayNetwork()
    if tree_type == "balanced":
        network.insert_balanced_BST(nodes)
    elif tree_type == "frequency":
        network.insert_frequency_BST(nodes)
    elif tree_type == "shuffled":
        network.insert_shuffled_BST(nodes)
    elif tree_type == "optimal":
        network.insert_optimal_BST(nodes, node_weights)

    _, total_cost_splaynet, total_adjustment_cost_splaynet, total_routing_cost_splaynet = splaynet(network, sigma)
    splaynet_cost_results = [total_cost_splaynet, total_adjustment_cost_splaynet, total_routing_cost_splaynet]

    return splaynet_cost_results


def run_simulation_lazy(nodes, sigma, alpha_value, tree_type, node_weights):
    network = SplayNetwork()
    if tree_type == "balanced":
        network.insert_balanced_BST(nodes)
    elif tree_type == "frequency":
        network.insert_frequency_BST(nodes)
    elif tree_type == "shuffled":
        network.insert_shuffled_BST(nodes)
    elif tree_type == "optimal":
        network.insert_optimal_BST(nodes, node_weights)

    _, total_cost_lazy, total_adjustment_cost_lazy, total_routing_cost_lazy, requests_per_adjustment = lazy_splaynet(
        network, sigma,
        alpha=alpha_value)
    lazy_cost_results = [total_cost_lazy, total_adjustment_cost_lazy, total_routing_cost_lazy, requests_per_adjustment]
    return lazy_cost_results


def run_simulation_sliding(nodes, sigma, window_size, slide_offset, alpha, tree_type, node_weights):
    network = SplayNetwork()
    if tree_type == "balanced":
        network.insert_balanced_BST(nodes)
    elif tree_type == "frequency":
        network.insert_frequency_BST(nodes)
    elif tree_type == "shuffled":
        network.insert_shuffled_BST(nodes)
    elif tree_type == "optimal":
        network.insert_optimal_BST(nodes, node_weights)

    _, total_cost_sliding, total_adjustment_cost_sliding, total_routing_cost_sliding, requests_per_adjustment = variable_sliding_window_splaynet(
        initial_topology=network, communication_sq=sigma,
        window_size=window_size, slide_offset=slide_offset, alpha=alpha)
    sliding_cost_results = [total_cost_sliding, total_adjustment_cost_sliding, total_routing_cost_sliding,
                            requests_per_adjustment]
    return sliding_cost_results


def run_simulation_sliding_no_reset(nodes, sigma, window_size, slide_offset, alpha, tree_type, node_weights):
    network = SplayNetwork()
    if tree_type == 'balanced':
        network.insert_balanced_BST(nodes)
    elif tree_type == 'frequency':
        network.insert_frequency_BST(nodes)
    elif tree_type == "shuffled":
        network.insert_shuffled_BST(nodes)
    elif tree_type == "optimal":
        network.insert_optimal_BST(nodes, node_weights)

    _, total_cost_sliding, total_adjustment_cost_sliding, total_routing_cost_sliding, requests_per_adjustment = variable_sliding_window_splaynet_no_reset(
        initial_topology=network, communication_sq=sigma,
        window_size=window_size, slide_offset=slide_offset, alpha=alpha)
    sliding_cost_results = [total_cost_sliding, total_adjustment_cost_sliding, total_routing_cost_sliding,
                            requests_per_adjustment]
    return sliding_cost_results


# Result calculating functions - Outputs are stored as csv for easier access for the plotting operations
def compute_temporal_results(tau_list, network_size_list, data_folder, lazy_splaynet_alphas, sliding_window_sizes,
                             sliding_offset_percentage_list, tree_type, ratio_percentages):
    node_weights = None
    combinations = []
    columns = ["network_size", "tau", "algorithm", "adjustment_cost", "routing_cost", "total_cost", "alpha",
               "window_size", "sliding_offset", "request_counter", "requests_per_adjustment"]
    results_df = pd.DataFrame(columns=columns)

    for tau in tau_list:
        for network_size in network_size_list:
            combinations.append(f"{tau}_{network_size}")

    # Ausgabe der Kombinationen
    for combination in combinations:
        tau = float(combination.split("_")[0])
        network_size = int(combination.split("_")[1])

        # sigma, nodes = csv_to_sequence("data/sample.csv")
        current_file = f"{data_folder}/data_{combination}.csv"
        if tree_type == "balanced":
            sigma, nodes = csv_to_sequence(current_file)
        elif tree_type == "frequency":
            sigma, nodes = csv_to_frequency_sequence(current_file)
        elif tree_type == "shuffled":
            sigma, nodes = csv_to_shuffled_sequence(current_file)
        elif tree_type == "optimal":
            sigma, nodes, node_weights = csv_to_optimal_sequence(current_file)

        print(f"Processing dataset for tau={tau} with {network_size} nodes.")

        print("Running SplayNet algorithm...")
        print(node_weights)
        results_splaynet = run_simulation_splaynet(nodes, sigma, tree_type, node_weights)
        # push to df
        results_df.loc[len(results_df)] = [network_size, tau, "splaynet", results_splaynet[1], results_splaynet[2],
                                           results_splaynet[0], None, None, None, None, None]

        for alpha_value in lazy_splaynet_alphas:
            print(f"Running Lazy SplayNet algorithm... with alpha={alpha_value}")
            results_lazy = run_simulation_lazy(nodes, sigma, alpha_value, tree_type,node_weights)
            lazy_rpa = sum(results_lazy[3]) / len(results_lazy[3])
            # push to df
            results_df.loc[len(results_df)] = [network_size, tau, "lazy", results_lazy[1], results_lazy[2],
                                               results_lazy[0], alpha_value, None, None, results_lazy[3], lazy_rpa]

            sliding_window_sizes = [int(alpha_value * p) for p in ratio_percentages]

            for window_size in sliding_window_sizes:
                sliding_offset_list = [int(window_size * p) for p in sliding_offset_percentage_list]
                for slide_offset in sliding_offset_list:
                    print(
                        f"Running SlidingWindow algorithm... with window size={window_size} and slide_offset={slide_offset}")
                    results_sliding = run_simulation_sliding(nodes, sigma, window_size, slide_offset, alpha_value,
                                                             tree_type, node_weights)
                    sliding_rpa = sum(results_sliding[3]) / len(results_sliding[3])
                    results_df.loc[len(results_df)] = [network_size, tau, "sliding", results_sliding[1],
                                                       results_sliding[2],
                                                       results_sliding[0], alpha_value, window_size, slide_offset,
                                                       results_sliding[3], sliding_rpa]

                    print(
                        f"Running SlidingWindow no reset algorithm... with window size={window_size} and slide_offset={slide_offset}")
                    results_sliding_no_reset = run_simulation_sliding_no_reset(nodes, sigma, window_size, slide_offset,
                                                                               alpha_value, tree_type, node_weights)
                    sliding_no_reset_rpa = sum(results_sliding_no_reset[3]) / len(results_sliding_no_reset[3])
                    results_df.loc[len(results_df)] = [network_size, tau, "sliding_no_reset",
                                                       results_sliding_no_reset[1],
                                                       results_sliding_no_reset[2], results_sliding_no_reset[0],
                                                       alpha_value, window_size, slide_offset,
                                                       results_sliding_no_reset[3], sliding_no_reset_rpa]

    output_path = f"output/{tree_type}_{add_timestamp("temporal_results")}.csv"
    results_df.to_csv(output_path)
    return output_path


def compute_spatial_results(zipf_list, network_size_list, data_folder, lazy_splaynet_alphas, sliding_window_sizes,
                            sliding_offset_percentage_list, tree_type, ratio_percentages):
    node_weights = None
    combinations = []
    columns = ["network_size", "zipf", "algorithm", "adjustment_cost", "routing_cost", "total_cost", "alpha",
               "window_size", "sliding_offset", "request_counter", "requests_per_adjustment"]
    results_df = pd.DataFrame(columns=columns)

    for zipf in zipf_list:
        for network_size in network_size_list:
            combinations.append(f"{zipf}_{network_size}")

    # Ausgabe der Kombinationen
    for combination in combinations:
        zipf = float(combination.split("_")[0])
        network_size = int(combination.split("_")[1])

        current_file = f"{data_folder}/data_{combination}.csv"
        if tree_type == "balanced":
            sigma, nodes = csv_to_sequence(current_file)
        elif tree_type == "frequency":
            sigma, nodes = csv_to_frequency_sequence(current_file)
        elif tree_type == "shuffled":
            sigma, nodes = csv_to_shuffled_sequence(current_file)
        elif tree_type == "optimal":
            sigma, nodes, node_weights = csv_to_optimal_sequence(current_file)

        print(f"Processing dataset for zipf={zipf} with {network_size} nodes.")

        print("Running SplayNet algorithm...")
        results_splaynet = run_simulation_splaynet(nodes, sigma, tree_type, node_weights)
        # push to df
        results_df.loc[len(results_df)] = [network_size, zipf, "splaynet", results_splaynet[1], results_splaynet[2],
                                           results_splaynet[0], None, None, None, None, None]

        for alpha_value in lazy_splaynet_alphas:
            print(f"Running Lazy SplayNet algorithm... with alpha={alpha_value}")
            results_lazy = run_simulation_lazy(nodes, sigma, alpha_value, tree_type, node_weights)
            req_per_adjustment = sum(results_lazy[3]) / len(results_lazy[3])
            # push to df
            results_df.loc[len(results_df)] = [network_size, zipf, "lazy", results_lazy[1], results_lazy[2],
                                               results_lazy[0], alpha_value, None, None, results_lazy[3],
                                               req_per_adjustment]

            sliding_window_sizes = [int(alpha_value * p) for p in ratio_percentages]
            for window_size in sliding_window_sizes:
                sliding_offset_list = [int(window_size * p) for p in sliding_offset_percentage_list]
                for slide_offset in sliding_offset_list:
                    print(
                        f"Running SlidingWindow algorithm... with alpha={alpha_value}, window size={window_size} and slide_offset={slide_offset}")
                    results_sliding = run_simulation_sliding(nodes, sigma, window_size, slide_offset, alpha_value,
                                                             tree_type, node_weights)
                    sliding_rpa = sum(results_sliding[3]) / len(results_sliding[3])
                    results_df.loc[len(results_df)] = [network_size, zipf, "sliding", results_sliding[1],
                                                       results_sliding[2], results_sliding[0], alpha_value, window_size,
                                                       slide_offset, results_sliding[3], sliding_rpa]

                    print(
                        f"Running SlidingWindow no reset algorithm... with alpha={alpha_value}, window size={window_size} and slide_offset={slide_offset}")
                    results_sliding_no_reset = run_simulation_sliding_no_reset(nodes, sigma, window_size, slide_offset,
                                                                               alpha_value, tree_type, node_weights)
                    sliding_no_reset_rpa = sum(results_sliding_no_reset[3]) / len(results_sliding_no_reset[3])
                    results_df.loc[len(results_df)] = [network_size, zipf, "sliding_no_reset",
                                                       results_sliding_no_reset[1],
                                                       results_sliding_no_reset[2], results_sliding_no_reset[0],
                                                       alpha_value, window_size, slide_offset,
                                                       results_sliding_no_reset[3], sliding_no_reset_rpa]

    output_path = f"output/{tree_type}_{add_timestamp("spatial_results")}.csv"
    results_df.to_csv(output_path)
    return output_path


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

    plt.legend()

    plt.grid(True)

    if log_scale_bool:
        plt.yscale('log')

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

    plt.legend()

    # Show grid
    plt.grid(True)
    type_text = "temporal" if zipf_or_tau == "tau" else "spatial"

    # Save plot with a timestamp in the filename
    plt.savefig(f"output/{type_text}_x_{zipf_or_tau}_y_total_cost_network_size_{network_size}{add_timestamp('')}.png")

    plt.show()


def plot_x_zipf_or_tau_y_average_cost(df, network_size):
    zipf_or_tau = "zipf" if "zipf" in df.columns else "tau"
    filtered_df = df[df['network_size'] == network_size]

    if filtered_df.empty:
        print(f"No data available for network size {network_size}")
        return

    avg_df = filtered_df.groupby([zipf_or_tau, 'algorithm']).mean().reset_index()

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

    avg_df = df.groupby(['network_size', 'algorithm']).mean().reset_index()

    plt.figure(figsize=(10, 6))

    for algorithm in algorithms:
        filtered_df = avg_df[avg_df['algorithm'] == algorithm]
        plt.plot(filtered_df['network_size'], filtered_df['total_cost'], label=algorithm.capitalize(), marker='o')

    plt.xlabel('Network Size')
    plt.ylabel('Average Total Cost')
    plt.title('Average Total Cost Comparison')

    plt.legend()

    plt.grid(True)

    type_text = "temporal" if zipf_or_tau == "tau" else "spatial"
    plt.savefig(
        f"output/{type_text}_x_network_size_y_average_cost_with_algorithms_{'_'.join(algorithms)}_{add_timestamp('')}.png")

    plt.show()


def plot_cost_ratio(df, network_size, cost_type1, cost_type2, algorithms):

    filtered_df = df[df['network_size'] == network_size]

    if filtered_df.empty:
        print(f"No data available for network size {network_size}")
        return

    plt.figure(figsize=(10, 6))

    for algorithm in algorithms:
        algorithm_df = filtered_df[filtered_df['algorithm'] == algorithm]

        if not algorithm_df.empty:

            cost_ratio = algorithm_df[cost_type1] / algorithm_df[cost_type2]

            plt.plot(algorithm_df['tau'], cost_ratio, label=f'{algorithm} {cost_type1}/{cost_type2}', marker='o')

    plt.xlabel('Tau')
    plt.ylabel(f'{cost_type1}/{cost_type2}')
    plt.title(f'Cost Ratio ({cost_type1}/{cost_type2}) vs Tau for Network Size {network_size}')

    plt.yscale('log')

    plt.grid(True)

    plt.legend()
    plt.show()


def compute_window_alpha_ratio(df):
    df['ratio_alpha_windowsize'] = df['alpha'] / df['window_size']
    average_cost_by_ratio = df.groupby('ratio_alpha_windowsize')[
        ['adjustment_cost', 'routing_cost', 'total_cost']].mean().reset_index()
    return average_cost_by_ratio


def plot_requests_per_window(ratio_df):
    average_cost_by_ratio_scaled = ratio_df.copy()
    average_cost_by_ratio_scaled['adjustment_cost'] /= 1000
    average_cost_by_ratio_scaled['routing_cost'] /= 1000
    average_cost_by_ratio_scaled['total_cost'] /= 1000

    bar_width = 0.25

    r1 = np.arange(len(average_cost_by_ratio_scaled))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]

    plt.figure(figsize=(12, 6))
    plt.bar(r1, average_cost_by_ratio_scaled['adjustment_cost'], color='b', width=bar_width, edgecolor='grey',
            label='Adjustment Cost')
    plt.bar(r2, average_cost_by_ratio_scaled['routing_cost'], color='g', width=bar_width, edgecolor='grey',
            label='Routing Cost')
    plt.bar(r3, average_cost_by_ratio_scaled['total_cost'], color='r', width=bar_width, edgecolor='grey',
            label='Total Cost')

    plt.xlabel('Ratio Alpha Window Size', fontweight='bold')
    plt.ylabel('Cost (in Thousands)', fontweight='bold')
    plt.title('Cost Metrics by Ratio Alpha Window Size')

    plt.xticks([r + bar_width for r in range(len(average_cost_by_ratio_scaled))],
               average_cost_by_ratio_scaled['ratio_alpha_windowsize'])

    plt.legend()

    plt.ylim(400, 1500)  # Adjust these values as needed

    plt.show()


def plot_requests_per_window_seperate_costs(ratio_df):
    average_cost_by_ratio_scaled = ratio_df.copy()
    average_cost_by_ratio_scaled['adjustment_cost'] /= 1000
    average_cost_by_ratio_scaled['routing_cost'] /= 1000
    average_cost_by_ratio_scaled['total_cost'] /= 1000

    bar_width = 0.25

    r = np.arange(len(average_cost_by_ratio_scaled))

    plt.figure(figsize=(12, 6))
    plt.bar(r, average_cost_by_ratio_scaled['adjustment_cost'], color='b', width=bar_width, edgecolor='grey')
    plt.xlabel('Ratio Alpha Window Size', fontweight='bold')
    plt.ylabel('Adjustment Cost (in Thousands)', fontweight='bold')
    plt.title('Adjustment Cost by Ratio Alpha Window Size')
    plt.xticks(r, average_cost_by_ratio_scaled['ratio_alpha_windowsize'])
    plt.ylim(470, 490)  # Adjust these values as needed
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.bar(r, average_cost_by_ratio_scaled['routing_cost'], color='g', width=bar_width, edgecolor='grey')
    plt.xlabel('Ratio Alpha Window Size', fontweight='bold')
    plt.ylabel('Routing Cost (in Thousands)', fontweight='bold')
    plt.title('Routing Cost by Ratio Alpha Window Size')
    plt.xticks(r, average_cost_by_ratio_scaled['ratio_alpha_windowsize'])
    plt.ylim(480, 510)  # Adjust these values as needed
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.bar(r, average_cost_by_ratio_scaled['total_cost'], color='r', width=bar_width, edgecolor='grey')
    plt.xlabel('Ratio Alpha Window Size', fontweight='bold')
    plt.ylabel('Total Cost (in Thousands)', fontweight='bold')
    plt.title('Total Cost by Ratio Alpha Window Size')
    plt.xticks(r, average_cost_by_ratio_scaled['ratio_alpha_windowsize'])
    plt.ylim(960, 1500)  # Adjust these values as needed
    plt.show()


def plot_spatial_req_per_adjustments_for_windows_and_alpha(df_spatial):
    sliding_data_spatial = df_spatial[df_spatial['algorithm'] == 'sliding']

    sliding_data_spatial['alpha/window_size'] = sliding_data_spatial['alpha'] / sliding_data_spatial['window_size']

    # Skalierung der Kostenwerte auf Tausendstel
    sliding_data_spatial['total_cost'] /= 1000
    sliding_data_spatial['adjustment_cost'] /= 1000
    sliding_data_spatial['routing_cost'] /= 1000
    sliding_data_spatial['requests_per_adjustment'] /= 1000

    # Definieren der X- und Y-Variablen
    x_vars = ['window_size', 'alpha/window_size']
    y_vars = ['total_cost', 'adjustment_cost', 'routing_cost', 'requests_per_adjustment']

    # Erstellen des Grids
    g = sns.PairGrid(sliding_data_spatial, x_vars=x_vars, y_vars=y_vars, hue='zipf', palette='viridis')

    # Scatterplot für den unteren Teil des Grids
    g.map(sns.lineplot)

    # Setzen der X-Achse auf logarithmischen Maßstab
    for ax in g.axes.flatten():
        if ax.get_xlabel() in x_vars:
            ax.set_xscale('log')

    # Hinzufügen einer Legende
    g.add_legend()

    # Titel hinzufügen
    plt.suptitle('Grid of Metrics for Sliding Algorithm with Zipf as Hue', y=1.02)

    # Anzeigen des Plots
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
    ''' Algorithm specific variables '''
    # Lazy SplayNet
    # threshold_alpha_list = [100, 250, 500, 1000, 5000, 10000]
    threshold_alpha_list = [128, 256, 512, 1024, 2048, 4096, 8192]

    # Sliding Window
    # sliding_window_size_list = [50, 100, 250, 500, 1000]
    sliding_window_size_list = [100, 250, 500, 1000]
    ratio_percentages = [0.25, 0.50, 1, 2, 4]
    # sliding_offset_percentages = [0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 1.0]
    sliding_offset_percentages = [1.0]
    # sliding_offset_percentages = [0.1, 0.9, 1.0]
    # sliding_offset_percentages = [0.25, 0.5, 0.75, 1.0]

    ''' Bool values and variables, that determine which parts of the workflow are executed '''
    # Generator booleans
    generate_spatial = False  # Set variables to True, if the datasets should be generated
    generate_temporal = False

    # Compute simulation booleans
    run_simulation = True

    # Plot graphs boolean
    plot = False
    plot_temporal = False
    plot_spatial = False
    log_scale = True  # True if results should be plotted with logarithmic scale
    average = True  # True if average cost should be plotted
    slice_temporal = ([None], [None])  # Variable to slice result as wished: ([network_size_slices], [parameter_slices])
    slice_spatial = ([None], [None])

    ''' Sequence parameters - can be adjusted as desired '''
    # all values
    # tau_parameters = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # zipf_parameters = [1.1, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]

    # testing values for test simulations
    # tau_parameters = [0.1]
    # zipf_parameters = [1.1]

    # high tau, low zipf - good performance estimated
    # tau_parameters = [0.1, 0.7, 0.8, 0.9]
    # zipf_parameters = [1.1, 1.25, 1.5, 2.5]

    # final values
    tau_parameters = [0.1, 0.3, 0.5, 0.7, 0.9]
    zipf_parameters = [1.1, 1.5, 2.0, 2.5]

    node_ranges = [1000]  # can either a single value [size] or as a tuple (start, stop, step) e.g. [100] (100, 1100, 100) (200, 1200, 200)
    sequence_length = 10 ** 6
    tree_type = "shuffled"  # options are "balanced", "frequency", "shuffled", "optimal"

    if len(node_ranges) == 1:
        size_list = [node_ranges[0]]
    elif 1 < len(node_ranges) < 4:
        size_list = list(range(*node_ranges))
    else:
        print("Invalid node range")
        size_list = []

    ''' Workflow '''
    # Generate communication sequence datasets
    if generate_spatial:
        print("Generating communication sequences for different spatial localities ...")
        spatial_sequences_folder = generate_sequence('spatial', size_list, zipf_parameters, sequence_length)
        print(f"Completed. Result(s) can be found in the folder {spatial_sequences_folder}. \n")
    else:
        spatial_sequences_folder = 'spatial_data_10_5'
        print(f"Communication requests are taken from the folder {spatial_sequences_folder}.\n")

    if generate_temporal:
        print("Generating temporal sequences for different temporal localities ...")
        temporal_sequences_folder = generate_sequence('temporal', size_list, tau_parameters, sequence_length)
        print(f"Completed. Result(s) can be found in the folder {temporal_sequences_folder}.\n")
    else:
        temporal_sequences_folder = 'temporal_data_10_5'
        print(f"Communication requests are taken from the folder '{temporal_sequences_folder}'\n.")

    # Run simulations on datasets - results are saved in a csv (see path) and can be used multiple times
    if run_simulation:
        '''test_main()'''  # runs simulation of sample.csv
        '''real_data_main(log_scale, average)'''  # runs simulations on all real traces in /csv
        temporal_output_path = compute_temporal_results(tau_parameters, size_list, temporal_sequences_folder,
                                                        lazy_splaynet_alphas=threshold_alpha_list,
                                                        sliding_window_sizes=sliding_window_size_list,
                                                        sliding_offset_percentage_list=sliding_offset_percentages,
                                                        tree_type=tree_type, ratio_percentages=ratio_percentages)

        spatial_output_path = compute_spatial_results(zipf_parameters, size_list, spatial_sequences_folder,
                                                      lazy_splaynet_alphas=threshold_alpha_list,
                                                      sliding_window_sizes=sliding_window_size_list,
                                                      sliding_offset_percentage_list=sliding_offset_percentages,
                                                      tree_type=tree_type, ratio_percentages=ratio_percentages)

    else:
        # Add path of desired simulation
        temporal_output_path = "output/frequency_temporal_results_20240719_194519.csv"
        spatial_output_path = "output/frequency_spatial_results_20240719_214707.csv"

        # temporal_output_path = "output/temporal_results_20240529_210645.csv"
        # spatial_output_path = "output/spatial_results_20240529_231840.csv"

        # temporal_output_path = "output/temporal_results_20240603_161420.csv"
        # spatial_output_path = "output/spatial_results_20240603_162414.csv"

        # temporal_output_path = "output/frequency_temporal_results_20240630_200238.csv"
        # spatial_output_path = "output/frequency_spatial_results_20240630_203028.csv"


    temporal_df = pd.read_csv(temporal_output_path)
    spatial_df = pd.read_csv(spatial_output_path)

    # Evaluate and generate plots
    if plot:
        # Remove comment symbol to execute desired plotting function

        # Plot average request per window (sorted by ratio of alpha and windowsize)
        # plot_spatial_req_per_adjustments_for_windows_and_alpha(spatial_df)
        #
        temporal_ratio = compute_window_alpha_ratio(temporal_df)
        plot_requests_per_window(temporal_ratio)
        plot_requests_per_window_seperate_costs(temporal_ratio)

        spatial_ratio = compute_window_alpha_ratio(spatial_df)
        plot_requests_per_window(spatial_ratio)
        plot_requests_per_window_seperate_costs(spatial_ratio)
        #
        # # Evaluate performance of communication sequences of different localities
        # plot_x_network_size_y_total_cost(temporal_df, log_scale)
        # plot_x_network_size_y_total_cost(temporal_df, not log_scale)
        #
        # if average:
        #     plot_x_network_size_y_average_cost(temporal_df, log_scale)
        #     plot_x_network_size_y_average_cost(temporal_df, not log_scale)
        #
        # plot_x_zipf_or_tau_y_total_cost(temporal_df, 1000)
        # plot_x_zipf_or_tau_y_total_cost(temporal_df, 200)
        # plot_x_zipf_or_tau_y_total_cost(temporal_df, 300)
        # plot_x_zipf_or_tau_y_total_cost(temporal_df, 400)
        #
        # plot_x_zipf_or_tau_y_average_cost(temporal_df, 1000)
        #
        # plot_avg_cost_comparison(temporal_df, ["lazy", "sliding", "sliding_no_reset"])
        # plot_avg_cost_comparison(temporal_df, ["sliding", "sliding_no_reset"])
        # plot_cost_ratio(temporal_df, 100, "total_cost_lazy", "total_cost_splaynet")

        # plot_cost_ratio(temporal_df, 1000, "total_cost", "routing_cost", ["lazy", "sliding", "sliding_no_reset"])

    #### ENDE

    # # plot_cost_ratio(temporal_df, 100, "total_cost_lazy", "total_cost_splaynet")
    #
    # # plot_cost_ratio(temporal_df, 1000, "total_cost", "routing_cost", ["lazy", "sliding", "sliding_no_reset"])
    #
    # # Spatial
    # plot_x_network_size_y_total_cost(spatial_df, log_scale)
    # plot_x_network_size_y_total_cost(spatial_df, not log_scale)
    #
    # if average:
    #     plot_x_network_size_y_average_cost(spatial_df, log_scale)
    #     plot_x_network_size_y_average_cost(spatial_df, not log_scale)
    #
    # plot_x_zipf_or_tau_y_total_cost(spatial_df, 1000)
    # plot_x_zipf_or_tau_y_average_cost(spatial_df, 1000)
    #
    # plot_avg_cost_comparison(spatial_df, ["lazy", "sliding", "sliding_no_reset"])
    # plot_avg_cost_comparison(spatial_df, ["sliding", "sliding_no_reset"])

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
    #            output_name = os.path.join(output_folder, f"scatter_{os.path.basename(csv_file)[:-4]}_{timestamp}.png")
    #             plt.savefig(output_name)
