from topology.Node import Node
from topology.SplayNetworkOnline import SplayNet
import copy
import matplotlib.pyplot as plt


def lsn(topology, communication_sq, delta):
    """
    HIER SOLLTE EIN TEXT STEHEN DARÜBER WAS DER ALGORITHMUS MACHT
    :param
        initial_graph: HIER SOLLTE EIN TEXT STEHEN DARÜBER WAS DER PARAMETER IST, BEDEUTET UND MACHT
        communication_sq: HIER SOLLTE EIN TEXT STEHEN DARÜBER WAS DER PARAMETER IST, BEDEUTET UND MACHT
        delta: Threshold value
    :return:
    """
    costs = []
    total_cost = 0
    virtual_topology = copy.deepcopy(topology)
    printcounter = 1

    for request in communication_sq:
        a = request.get_source()
        b = request.get_destination()

        virtual_topology.commute(a, b)
        current_cost = virtual_topology.getServiceCost() + virtual_topology.getRotationCost()
        total_cost += current_cost
        # costs.append(total_cost)

        if total_cost >= delta:
            topology = copy.deepcopy(virtual_topology)
            printcounter += 1
            total_cost = 0

    # # Kosten plotten
    # plt.figure(figsize=(10, 5))
    # plt.plot(costs, label='Kumulierte Kosten über die Zeit')
    # plt.xlabel('Anzahl der Kommunikationsanfragen')
    # plt.ylabel('Kumulierte Kosten')
    # plt.title('Kostenanalyse der SplayNet Operationen')
    # plt.legend()
    # plt.savefig('Kostenanalyse.png')
    # plt.close()

    return topology


def lsn_online(topology, communication_sq, delta):
    costs = []  # Liste zum Speichern der Kosten nach jeder Operation
    total_cost = 0
    virtual_topology = copy.deepcopy(topology)
    printcounter = 1

    for request in communication_sq:
        a = request.get_source()
        b = request.get_destination()

        virtual_topology.commute(a, b)
        current_cost = virtual_topology.getServiceCost() + virtual_topology.getRotationCost()
        total_cost += current_cost
        costs.append(total_cost)  # Kosten hinzufügen

        if total_cost >= delta:
            topology = copy.deepcopy(virtual_topology)
            printcounter += 1
            total_cost = 0  # Kosten zurücksetzen

    # # Kosten plotten
    # plt.figure(figsize=(10, 5))
    # plt.plot(costs, label='Kumulierte Kosten über die Zeit')
    # plt.xlabel('Anzahl der Kommunikationsanfragen')
    # plt.ylabel('Kumulierte Kosten')
    # plt.title('Kostenanalyse der SplayNet Operationen')
    # plt.legend()
    # plt.savefig('Kostenanalyse.png')
    # plt.close()


def simplified_lsn(topology, communication_sq, width):
    cost = 0
    virtual_topology = topology

    for i, request in enumerate(communication_sq):
        a = request.get_source()
        b = request.get_destination()
        # virtual_topology = topology

        virtual_topology.route(a, b)
        cost += virtual_topology.routing_cost
        if i % width == 0:
            topology = virtual_topology

    return topology


def sliding_window_lsn(G_0, communication_sq, width, slide, threshold):
    actual_topology = G_0
    cost = 0
    for i, request in enumerate(communication_sq):
        a = request.get_source()
        b = request.get_destination()
        actual_topology.route(a, b)
        cost += actual_topology.routing_cost
        if i % slide == 0 and i != slide:
            resulting_topology = actual_topology
            # resulting_topology.print_tree(resulting_topology.root)
            actual_topology = copy.deepcopy(G_0)
