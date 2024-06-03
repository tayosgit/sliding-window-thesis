from topology.Node import Node


# class SplayNetwork:
#     def __init__(self):
#             self.root = None
#             self.routing_cost = 0
#             self.adjustment_cost = 0
#
#     def insert_balanced_BST(self, node_list: [int]):
#         node_list.sort()
#         k = len(node_list) // 2
#         self.root = Node(node_list[k])
#         self.root.left = self.insertion_iteration(node_list[:k], self.root)
#         self.root.right = self.insertion_iteration(node_list[k + 1:], self.root)
#
#     def insertion_iteration(self, node_list: [int], parent: Node):
#         if len(node_list) == 0:
#             return None
#         k = len(node_list) // 2
#         new_node = Node(node_list[k])
#         new_node.parent = parent
#         new_node.left = self.insertion_iteration(node_list[:k], new_node)
#         new_node.right = self.insertion_iteration(node_list[k + 1:], new_node)
#         return new_node

class SplayNetwork:

    def __init__(self):
        self.root = None
        self.routing_cost = 0
        self.adjustment_cost = 0

    def get_adjustment_cost(self):
        return self.adjustment_cost

    def get_routing_cost(self):
        return self.routing_cost

    def increase_adjustment_cost(self, cost: int):
        self.adjustment_cost += cost

    def increase_routing_cost(self, cost: int):
        if cost < 0:
            raise Exception("distance < 0")
        if cost == 0:
            raise Exception("distance = 0")
        self.routing_cost += cost

    def insert_balanced_BST(self, node_list: [int]):
        node_list.sort()
        k = len(node_list) // 2
        self.root = Node(node_list[k])
        self.root.left = self.insertion_iteration(node_list[:k], self.root)
        self.root.right = self.insertion_iteration(node_list[k + 1:], self.root)

    def insertion_iteration(self, node_list: [int], parent: Node):
        if len(node_list) == 0:
            return None
        k = len(node_list) // 2
        new_node = Node(node_list[k])
        new_node.parent = parent
        new_node.left = self.insertion_iteration(node_list[:k], new_node)
        new_node.right = self.insertion_iteration(node_list[k + 1:], new_node)
        return new_node

    def commute(self, u: int, v: int):
        node_u = u
        node_v = v
        common_ancestor = self.find_LCA(node_u, node_v)
        distance = self.calculate_distance(node_u, node_v)
        # print(f"u: {node_u}, v: {node_v}, distance: {distance}")

        if common_ancestor.key == node_v:
            node_u, node_v = node_v, node_u
        parent_CA = common_ancestor.parent or common_ancestor
        if parent_CA.key > common_ancestor.key:
            new_LCA = self.splay_wrapper(common_ancestor, node_u)
        elif parent_CA.key < common_ancestor.key:
            new_LCA = self.splay_wrapper(common_ancestor, node_u)
        else:
            new_LCA = self.splay_wrapper(self.root, node_u)
        # if node_u == node_v:
        #     raise Exception("gleiche Knoten kommunizieren")
        if node_u > node_v:
            self.splay_wrapper(new_LCA.left, node_v)
        if node_u < node_v:
            self.splay_wrapper(new_LCA.right, node_v)
        # distance = self.calculate_distance(node_u, node_v)
        self.increase_routing_cost(distance)

    def find_LCA(self, u: int, v: int) -> Node:
        node = self.root
        while node and ((u > node.key and v > node.key) or (u < node.key and v < node.key)):
            if u > node.key:
                node = node.right
            else:
                node = node.left
        assert node is not None
        return node

    def splay_wrapper(self, h: Node, key: int) -> Node:
        if not h:
            raise Exception("Node in Splay() does not exist")
        node = h
        while node:
            if node.key > key:
                node = node.left
            elif node.key < key:
                node = node.right
            elif node.key == key:
                self.splay_up(h, node)
                return node
        raise Exception("Node in Splay() not found")

    def splay_up(self, h: Node, k: Node) -> Node:
        if h == k:
            return k
        elif k.parent == h:
            if h.left == k:
                self.rotate_right(h)
            elif h.right == k:
                self.rotate_left(h)
            else:
                raise Exception("H should be parent, but not has k as child")
            return k
        else:
            found = h == k.parent.parent
            if k.parent.parent.right == k.parent:
                if k.parent.right == k:
                    self.rotate_left(k.parent.parent)
                elif k.parent.left == k:
                    self.rotate_right(k.parent)
                else:
                    raise Exception("k.p has not k as child")
                self.rotate_left(k.parent)
            elif k.parent.parent.left == k.parent:
                if k.parent.right == k:
                    self.rotate_left(k.parent)
                elif k.parent.left == k:
                    self.rotate_right(k.parent.parent)
                else:
                    raise Exception("k.p has not k as child")
                self.rotate_right(k.parent)
            else:
                raise Exception("k.p.p has not k.p as child")
            if found:
                return k
        return self.splay_up(h, k)

    def rotate_right(self, h: Node) -> Node:
        if not h.left:
            raise Exception("kein linkes kind bei rechtsrotation")
        x = h.left
        h.left = x.right
        cost = 0
        if x.right:
            cost += 2
        if h.parent:
            cost += 2
        if h.left:
            h.left.parent = h
        x.right = h
        x.parent = h.parent
        h.parent = x
        if x.parent:
            if x.parent.left == h:
                x.parent.left = x
            elif x.parent.right == h:
                x.parent.right = x
            else:
                raise Exception("x.p hatte h nicht als kind")
        else:
            self.root = x
        cost += 2
        self.increase_adjustment_cost(cost)
        x.lastRightParent = h.lastRightParent
        h.lastLeftParent = x.key
        return x

    def rotate_left(self, h: Node) -> Node:
        if not h.right:
            raise Exception("kein rechtes kind bei linksrotation")
        x = h.right
        h.right = x.left
        cost = 0
        if x.left:
            cost += 2
        if h.parent:
            cost += 2
        if h.right:
            h.right.parent = h
        x.left = h
        x.parent = h.parent
        h.parent = x
        if x.parent:
            if x.parent.left == h:
                x.parent.left = x
            elif x.parent.right == h:
                x.parent.right = x
            else:
                raise Exception("x.p hatte h nicht als kind")
        else:
            self.root = x
        cost += 2
        self.increase_adjustment_cost(cost)
        x.lastLeftParent = h.lastLeftParent
        h.lastRightParent = x.key
        return x

    def calculate_distance(self, a, b):
        current = self.root
        # common_ancestor = None
        distance = 0
        while current is not None and ((a > current.key and b > current.key) or (a < current.key and b < current.key)):
            if a > current.key:
                current = current.right
            else:
                current = current.left
        common_ancestor = current
        # print(f"common_ancestor: {common_ancestor.key}")
        while current is not None and current.key != a and current.left is not None:
            if a < current.left.key:
                current = current.left
            else:
                current = current.right
            distance += 1
        current = common_ancestor
        # print(f"common_ancestor 2: {common_ancestor.key}")
        while current is not None and current.key != b:
            if b < current.key:
                current = current.left
            else:
                current = current.right
            distance += 1

        if common_ancestor.key == a or common_ancestor.key == b:
            distance += 1

        return distance

    def print_tree(self, node: Node, indent: int = 0, prefix: str = "Root"):
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
        else:
            print("Node is None")
