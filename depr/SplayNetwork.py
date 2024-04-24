from topology.Node import Node


class SplayNetwork:
    def __init__(self):
        # self.networkId = id
        self.root = None
        self.edges = None

        self.routing_cost = 0
        self.rotation_cost = 0  # TODO Überprüfen: Vielleicht gar nicht gebraucht?

    def __str__(self):
        return str(self)

    def print_tree(self, node, indent=0, prefix="Root"):
        """Druckt jeden Knoten des Baumes mit einer visuellen Darstellung der Eltern-Kind-Beziehungen."""
        if node is not None:
            # Drucke den rechten Teilbaum zuerst mit einem "\" um die Verbindung anzuzeigen
            if node.right:
                # self.print_tree(node.right, indent + 4, prefix="\\---")
                self.print_tree(node.right, indent + 4, prefix="/---")
            # Drucke den aktuellen Knoten: Einrückungen + Präfix + Knotenwert
            print(' ' * indent + prefix + str(node.key))

            # Drucke den linken Teilbaum zuletzt mit einem "/" um die Verbindung anzuzeigen
            if node.left:
                # self.print_tree(node.left, indent + 4, prefix="/---")
                self.print_tree(node.left, indent + 4, prefix="\\---")

    # Create balanced BST in network
    def build_network(self, node_list):
        node_list.sort()
        l = len(node_list) // 2
        self.root = Node(node_list[l])
        self.root.left = self.insertion_iteration(node_list[:l], self.root)
        self.root.right = self.insertion_iteration(node_list[l + 1:], self.root)
        print("Netzwerk erfolgreich erstellt!")

    def insertion_iteration(self, node_list, parent):
        if len(node_list) == 0:
            return None
        l = len(node_list) // 2
        new_node = Node(node_list[l])
        new_node.parent = parent
        new_node.left = self.insertion_iteration(node_list[:l], new_node)
        new_node.right = self.insertion_iteration(node_list[l + 1:], new_node)
        return new_node

    # Routing of communication requests
    def route(self, a, b):
        if a == b:
            raise Exception("Node trys to route to itself.")

        common_ancestor = self.find_lowest_common_ancestor(a, b)

        if common_ancestor.key == b:
            a, b = b, a

        parent_ca = common_ancestor.parent if common_ancestor.parent is not None else common_ancestor

        if parent_ca.key > common_ancestor.key or parent_ca.key < common_ancestor.key:
            new_ca = self.splay_wrapper(common_ancestor, a)
        else:
            new_ca = self.splay_wrapper(self.root, a)

        if a > b:
            self.splay_wrapper(new_ca.left, b)
        if a < b:
            self.splay_wrapper(new_ca.right, b)

    # Splay
    def splay_wrapper(self, node_a, key):
        if not node_a:
            raise Exception(f"Node {node_a} is not defined")
        current = node_a
        while current:
            if current.key > key:
                current = current.left
            elif current.key < key:
                current = current.right
            elif current.key == key:
                self.splay(node_a, current)
                return current
        raise Exception(f"Node {node_a} not found in Network.")

    def splay(self, node_a, node_b):
        """
        Splays node a to position of node b
        :param
            node_a: Node to be splayed
            node_b: Node to which position is to be splayed
        :return:
        """
        if node_a == node_b:  # Case 1: Nodes are identical
            return node_b

        elif node_a == node_b.parent:  # Case 2: Node a is parent of node b
            if node_a.left == node_b:  # Case 2.1: Node b is left child of node a
                self.rotate_right(node_a)
            elif node_a.right == node_b:  # Case 2.2: Node b is right child of node a
                self.rotate_left(node_a)
            else:
                raise Exception(
                    f"Invalid! Node {node_a} is parent of node {node_b} but does not have {node_b} as a child")

        else:

            if node_b.parent.parent.right == node_b.parent:
                if node_b.parent.left == node_b:
                    self.rotate_right(node_b.parent.parent)
                elif node_b.parent.right == node_b:
                    self.rotate_left(node_b.parent)
                else:
                    raise Exception(f"Parent {node_b.parent} has no child {node_b}")
                self.rotate_left(node_b.parent)
            elif node_b.parent.parent.left == node_b.parent:
                if node_b.parent.left == node_b:
                    self.rotate_right(node_b.parent.parent)
                elif node_b.parent.right == node_b:
                    self.rotate_left(node_b.parent)
                else:
                    raise Exception(f"Parent {node_b.parent} has no child {node_b}")
                self.rotate_right(node_b.parent)
            else:
                raise Exception(f"Grandparent {node_b.parent.parent} has no child{node_b.parent}")
            if at_root:
                return node_b
        return self.splay(node_a, node_b)

    # Helpers Splay
    # def find_lowest_common_ancestor(self, a, b):
    #     current = self.root
    #     while current is not None and
    #     ((a > current.key and b > current.key) or (a < current.key and b < current.key)):
    #         if a > current.key:
    #             current = current.right
    #         else:
    #             current = current.left
    #     assert current is not None
    #     return current

    def find_lowest_common_ancestor(self, a, b):
        current = self.root
        while (current is not None and ((a > current.key and b > current.key) or (a < current.key and b < current.key))):
            if a > current.key:
                current = current.right
            else:
                current = current.left
        assert current is not None
        return current



    def rotate_left(self, node):
        if not node.right:
            raise Exception("No right child found")

        cost = 0
        temp_node = node.right
        node.right = temp_node.left

        # TODO Kommentar für die Kosten hinzufügen
        if temp_node.left:
            cost += 2
        if node.parent:
            cost += 2
        if node.right:
            node.right.parent = node

        temp_node.left = node
        temp_node.parent = node.parent
        node.parent = temp_node

        if temp_node.parent:
            if temp_node.parent.left == node:
                temp_node.parent.left = temp_node
            elif temp_node.parent.right == node:
                temp_node.parent.right = temp_node
            else:
                raise Exception(f"Temp_node {temp_node} is not the parent of {node}")

        else:
            self.root = temp_node
        cost += 2
        self.update_rotation_cost(cost)  # Unklar, ob das nötig
        # Weitere Funktionaliäten für Kostenberechnung HIER
        return temp_node

    def rotate_right(self, node):
        if not node.left:
            raise Exception("No left child found")

        cost = 0
        temp_node = node.left
        node.left = temp_node.right

        # TODO Kommentar für die Kosten hinzufügen
        if temp_node.right:
            cost += 2
        if node.parent:
            cost += 2
        if node.left:
            node.left.parent = node

        temp_node.right = node
        temp_node.parent = node.parent
        node.parent = temp_node

        if temp_node.parent:
            if temp_node.parent.left == node:
                temp_node.parent.left = temp_node
            elif temp_node.parent.right == node:
                temp_node.parent.right = temp_node
            else:
                raise Exception(f"Temp_node {temp_node} is not the parent of {node}")

        else:
            self.root = temp_node
        cost += 2
        self.update_rotation_cost(cost)  # Unklar, ob das nötig
        # Weitere Funktionaliäten für Kostenberechnung HIER
        return temp_node

    # Helpers Cost
    def update_routing_cost(self, cost):
        self.routing_cost += cost

    def update_rotation_cost(self, cost):
        self.rotation_cost += cost

    def calculate_distance(self, a, b):
        current = self.root
        common_ancestor = None
        distance = 0
        while current is not None and ((a > current.key and b > current.key) or (a < current.key and b < current.key)):
            if a > current.key:
                current = current.right
            else:
                current = current.left
        common_ancestor = current
        while current is not None and current.key != a:
            if a < current.left:
                current = current.left
            else:
                current = current.right
            distance += 1
        current = common_ancestor
        while current is not None and current.key != b:
            if b < current.key:
                current = current.left
            else:
                current = current.right
            distance += 1

        return distance
