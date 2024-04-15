def lsn(topology, communication_sq, delta):
    """
    HIER SOLLTE EIN TEXT STEHEN DARÜBER WAS DER ALGORITHMUS MACHT
    :param
        initial_graph: HIER SOLLTE EIN TEXT STEHEN DARÜBER WAS DER PARAMETER IST, BEDEUTET UND MACHT
        communication_sq: HIER SOLLTE EIN TEXT STEHEN DARÜBER WAS DER PARAMETER IST, BEDEUTET UND MACHT
        delta: Threshold value
    :return:
    """
    cost = 0
    virtual_topology = topology

    for request in communication_sq:
        a = request.get_source()
        b = request.get_destination()
        # virtual_topology = topology

        virtual_topology.route(a, b)
        cost += virtual_topology.routing_cost
        if cost >= delta:
            topology = virtual_topology
            cost = 0

    return topology


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
            resulting_topology.print_tree(resulting_topology.root)
            actual_topology = G_0


