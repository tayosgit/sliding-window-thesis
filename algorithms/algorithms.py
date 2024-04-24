import copy
from topology.SplayNetwork import SplayNetwork
from topology.CommunicationRequest import CommunicationRequest


def splaynet(topology: SplayNetwork, communication_sq: [CommunicationRequest]) -> (SplayNetwork, int):

    for request in communication_sq:
        u = request.src
        v = request.dst

        topology.commute(u, v)

    total_cost = topology.get_routing_cost() + topology.get_adjustment_cost()

    return topology, total_cost


def lazy_splaynet(topology: SplayNetwork, communication_sq: [CommunicationRequest], alpha: int) -> (SplayNetwork, int):
    total_cost = 0
    accumulated_cost = 0
    virtual_topology = topology
    # virtual_topology = copy.deepcopy(topology)

    for request in communication_sq:
        u = request.src
        v = request.dst

        virtual_topology.commute(u, v)
        accumulated_cost = virtual_topology.get_routing_cost() + virtual_topology.get_adjustment_cost()

        if accumulated_cost > alpha:
            total_cost += virtual_topology.calculate_distance(u, v)
            # topology = copy.deepcopy(virtual_topology)
            topology = virtual_topology
            total_cost += 1
            accumulated_cost = 0
        else:
            total_cost += 2  # Für die virtuellen Berechnungen

    return topology, total_cost


def sliding_window_splaynet(initial_topology: SplayNetwork, communication_sq: [CommunicationRequest],
                            slide: int) -> (SplayNetwork, int):
    total_cost = 0
    virtual_topology = initial_topology
    resulting_topology = initial_topology
    # virtual_topology = copy.deepcopy(initial_topology)
    # resulting_topology = copy.deepcopy(initial_topology)

    for i, request in enumerate(communication_sq):
        u = request.src
        v = request.dst

        virtual_topology.commute(u, v)

        if i % slide == 0 and i != slide:  # width of sliding window
            resulting_topology = virtual_topology
            # resulting_topology = copy.deepcopy(virtual_topology)
            total_cost += 1 + resulting_topology.calculate_distance(u, v)
            virtual_topology = initial_topology
            # virtual_topology = copy.deepcopy(initial_topology)
        else:
            total_cost += 2  # Für die virtuellen Berechnungen

    return resulting_topology, total_cost
