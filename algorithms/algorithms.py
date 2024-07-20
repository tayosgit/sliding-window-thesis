import copy
from topology.SplayNetwork import SplayNetwork
from topology.CommunicationRequest import CommunicationRequest


# Helper functions
def window_iteration(topology: SplayNetwork, sequence_slice: [CommunicationRequest]):
    # Zählt die Male die das netzwerk angepasst wurde
    for request in sequence_slice:
        u = request.get_source()
        v = request.get_destination()
        topology.commute(u, v)
    total_adjustment_cost = topology.get_adjustment_cost()
    return topology, total_adjustment_cost


def generate_tuples(n, i):
    result = []
    start_index = 0
    while start_index < n:
        end_index = start_index + i
        result.append((start_index, end_index))
        start_index += i // 2
    return result


def sliding_window_splaynet_helper(initial_topology: SplayNetwork, communication_sq: [CommunicationRequest],
                                   slide_start: int, slide_end: int) -> (SplayNetwork, int):
    total_cost = 0
    virtual_topology = initial_topology
    resulting_topology = initial_topology

    for i, request in enumerate(communication_sq):
        u = request.get_source()
        v = request.get_destination()

        virtual_topology.commute(u, v)
        if i in range(slide_start, slide_end):  # width of sliding window
            resulting_topology = virtual_topology
            total_cost += 1 + resulting_topology.calculate_distance(u, v)
            virtual_topology = initial_topology
        else:
            total_cost += 2

    return resulting_topology, total_cost


# Algorithms (without NetworkX)
# SplayNet - for comparison
def splaynet(topology: SplayNetwork, communication_sq: [CommunicationRequest]) -> (SplayNetwork, int):
    # virtual_topology = copy.deepcopy(topology)

    for i, request in enumerate(communication_sq):
        u = request.get_source()
        v = request.get_destination()

        topology.commute(u, v)

    total_adjustment_cost = topology.get_adjustment_cost()
    total_routing_cost = topology.get_routing_cost()
    total_cost = total_routing_cost + total_adjustment_cost

    return topology, total_cost, total_adjustment_cost, total_routing_cost


# Lazy SplayNet - for comparison
def lazy_splaynet(topology: SplayNetwork, communication_sq: [CommunicationRequest], alpha: int) -> (SplayNetwork, int):
    request_counter_list = []
    request_counter = 0

    total_cost = 0
    total_adjustment_cost = 0
    total_routing_cost = 0
    epoch_cost = 0

    virtual_topology = copy.deepcopy(topology)

    accumulated_cost = 0
    previous_cost = 0

    for i, request in enumerate(communication_sq):
        u = request.get_source()
        v = request.get_destination()

        current_routing_cost = virtual_topology.calculate_distance(u, v)

        virtual_topology.commute(u, v)

        epoch_cost += current_routing_cost
        total_cost += current_routing_cost
        total_routing_cost += current_routing_cost

        request_counter += 1

        if epoch_cost >= alpha:
            total_cost += alpha  # Costs for the adjustment
            total_adjustment_cost += alpha
            topology = copy.deepcopy(virtual_topology)
            epoch_cost = 0

            request_counter_list.append(request_counter)
            request_counter = 0
    request_counter_list.append(request_counter)
    return topology, total_cost, total_adjustment_cost, total_routing_cost, request_counter_list


# Variable Sliding Window SplayNet - NEW APPROACH with fixed tree (reset)
def variable_sliding_window_splaynet(initial_topology, communication_sq, window_size, slide_offset, alpha):
    request_counter_list = []
    request_counter = 0

    total_cost = 0
    total_adjustment_cost = 0
    total_routing_cost = 0
    epoch_cost = 0

    virtual_topology = copy.deepcopy(initial_topology)
    resulting_topology = None

    for i, request in enumerate(communication_sq):
        u = request.get_source()
        v = request.get_destination()

        current_routing_cost = virtual_topology.calculate_distance(u, v)
        virtual_topology.commute(u, v)

        total_cost += current_routing_cost
        total_routing_cost += current_routing_cost
        epoch_cost += current_routing_cost

        request_counter += 1

        if epoch_cost >= alpha:
            resulting_topology = copy.deepcopy(initial_topology)  # reset to initial topology
            resulting_topology.adjustment_cost = 0
            # print(f"Adjustments are made based on communication_sq[{max(0, i - window_size)}: {i}]")
            resulting_topology, adjustment_cost = window_iteration(resulting_topology,
                                                                   communication_sq[max(0, i - window_size):i])
            total_cost += alpha
            total_adjustment_cost += alpha
            virtual_topology = copy.deepcopy(resulting_topology)
            epoch_cost = 0

            request_counter_list.append(request_counter)
            request_counter = 0

    request_counter_list.append(request_counter)

    return resulting_topology, total_cost, total_adjustment_cost, total_routing_cost, request_counter_list


# Variable Sliding Window SplayNet - NEW APPROACH without reset
def variable_sliding_window_splaynet_no_reset(initial_topology, communication_sq, window_size, slide_offset, alpha):
    request_counter_list = []
    request_counter = 0

    total_cost = 0
    total_adjustment_cost = 0
    total_routing_cost = 0
    epoch_cost = 0

    virtual_topology = copy.deepcopy(initial_topology)
    resulting_topology = copy.deepcopy(initial_topology)
    previous_routing_cost = 0

    for i, request in enumerate(communication_sq):
        u = request.get_source()
        v = request.get_destination()

        current_routing_cost = virtual_topology.calculate_distance(u, v)
        virtual_topology.commute(u, v)

        total_cost += current_routing_cost
        total_routing_cost += current_routing_cost
        epoch_cost += current_routing_cost

        request_counter += 1

        if epoch_cost >= alpha:
            resulting_topology.adjustment_cost = 0
            resulting_topology, current_adjustment_cost = window_iteration(resulting_topology,
                                                                           communication_sq[max(0, i - window_size):i])
            total_cost += alpha
            total_adjustment_cost += alpha
            virtual_topology = copy.deepcopy(resulting_topology)
            epoch_cost = 0

            request_counter_list.append(request_counter)
            request_counter = 0
    request_counter_list.append(request_counter)

    return resulting_topology, total_cost, total_adjustment_cost, total_routing_cost, request_counter_list


# Previous versions of Sliding Window (new approach)
def sliding_window_splaynet(initial_topology: SplayNetwork, communication_sq: [CommunicationRequest],
                            slide: int) -> (SplayNetwork, int):
    total_cost = 0
    virtual_topology = initial_topology
    resulting_topology = initial_topology

    for i, request in enumerate(communication_sq):
        u = request.get_source()
        v = request.get_destination()

        virtual_topology.commute(u, v)
        if i % slide == 0 and i != slide:  # width of sliding window
            resulting_topology = virtual_topology
            total_cost += 1 + resulting_topology.calculate_distance(u, v)
            virtual_topology = initial_topology
        else:
            total_cost += 2  # Für die virtuellen Berechnungen

    return resulting_topology, total_cost


def overlapping_sliding_window_splaynet(initial_topology: SplayNetwork, communication_sq: [CommunicationRequest],
                                        slide: int) -> (SplayNetwork, int):
    total_cost = 0
    windows = generate_tuples(len(communication_sq), slide)

    topologies = []
    for start_index, end_index in windows:
        resulting_topology_iteration, total_cost_iteration = sliding_window_splaynet_helper(initial_topology,
                                                                                            communication_sq,
                                                                                            start_index, end_index)
        topologies.append(resulting_topology_iteration)
        total_cost += total_cost_iteration

    return topologies, total_cost
