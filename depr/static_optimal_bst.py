from topology.Node import Node


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

    # Balanced BST
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

    # Frequency Based BST
    def insert_frequency_BST(self, node_list: [int]):
        for node in node_list:
            self.insert_node(node)

    def insert_node(self, key):
        if self.root is None:
            self.root = Node(key)
        else:
            self._insert(self.root, key)

    def _insert(self, current_node, key):
        if key < current_node.key:
            if current_node.left is None:
                current_node.left = Node(key)
                current_node.left.parent = current_node
                current_node.left.lastLeftParent = current_node.lastLeftParent
                current_node.left.lastRightParent = current_node.key
            else:
                self._insert(current_node.left, key)
        else:
            if current_node.right is None:
                current_node.right = Node(key)
                current_node.right.parent = current_node
                current_node.right.lastLeftParent = current_node.key
                current_node.right.lastRightParent = current_node.lastRightParent
            else:
                self._insert(current_node.right, key)

    # Randomly Shuffled BST
    def insert_shuffled_BST(self, node_list: [int]):
        for node in node_list:
            self.insert_node(node)

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
        if node_u > node_v:
            self.splay_wrapper(new_LCA.left, node_v)
        if node_u < node_v:
            self.splay_wrapper(new_LCA.right, node_v)
        # distance = self.calculate_distance(node_u, node_v)
        self.increase_routing_cost(distance)

    # Static Optimal BST

    def insert_optimal_BST(self, node_list: [int], node_weights: [[int]]):
        allNodes = [Node(key) for key in node_list]
        subtrees_cost, allRoots = self.calculate_optimal_BST(node_weights, len(allNodes))
        self.root = self.construct_optimal_BST_iterative(allNodes, allRoots, 0, len(allNodes) - 1)

    def calculate_optimal_BST(self, node_weights, n):
        subtrees_cost = [[0] * n for _ in range(n)]
        allRoots = [[0] * n for _ in range(n)]
        W = [[0] * n for _ in range(n)]

        # Improved calculation of W
        for i in range(n):
            for j in range(i, n):
                W[i][j] = node_weights[i][j] + (W[i][j - 1] if j > i else 0)

        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                subtrees_cost[i][j] = float('inf')
                for r in range(i, j + 1):
                    cost = (subtrees_cost[i][r - 1] if r > i else 0) + \
                           (subtrees_cost[r + 1][j] if r < j else 0) + \
                           W[i][j]
                    if cost < subtrees_cost[i][j]:
                        subtrees_cost[i][j] = cost
                        allRoots[i][j] = r

        return subtrees_cost, allRoots

    def construct_optimal_BST_iterative(self, allNodes, allRoots, beg, end):
        if beg > end:
            return None

        stack = [(beg, end, None, None)]
        visited = set()  # To track visited nodes and avoid infinite loops

        while stack:
            beg, end, parent, direction = stack.pop()
            if beg <= end:
                root_index = allRoots[beg][end]
                if (beg, end, root_index) in visited:
                    continue  # Skip already visited nodes
                visited.add((beg, end, root_index))

                root = allNodes[root_index]
                if parent is None:
                    self.root = root
                else:
                    if direction == "left":
                        parent.left = root
                    else:
                        parent.right = root

                # Debugging output
                print(f"Processed node with key: {root.key}, beg: {beg}, end: {end}, root_index: {root_index}")

                stack.append((root_index + 1, end, root, "right"))
                stack.append((beg, root_index - 1, root, "left"))

        return self.root
    def insert_optimal_BST(self, node_list: [int], node_weights: [[int]]):
        allNodes = [Node(key) for key in node_list]
        subtrees_cost, allRoots = self.calculate_optimal_BST(node_weights, len(allNodes))
        self.root = self.construct_optimal_BST(allNodes, allRoots, 0, len(allNodes) - 1)

    def calculate_optimal_BST(self, node_weights, n):
        subtrees_cost = [[0] * n for _ in range(n)]
        allRoots = [[0] * n for _ in range(n)]
        W = [[0] * n for _ in range(n)]

        for i in range(n):
            for j in range(i, n):
                W[i][j] = node_weights[i][j]
                if i != j:
                    W[i][j] += W[i][j - 1] if j > 0 else 0

        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                subtrees_cost[i][j] = float('inf')
                for r in range(i, j + 1):
                    cost = (subtrees_cost[i][r - 1] if r > i else 0) + \
                           (subtrees_cost[r + 1][j] if r < j else 0) + \
                           W[i][j]
                    if cost < subtrees_cost[i][j]:
                        subtrees_cost[i][j] = cost
                        allRoots[i][j] = r

        return subtrees_cost, allRoots

    def construct_optimal_BST(self, allNodes, allRoots, beg, end):
        if beg > end:
            return None
        root_index = allRoots[beg][end]
        root = allNodes[root_index]
        print(f"Constructing node with key: {root.key}, beg: {beg}, end: {end}, root_index: {root_index}")
        if beg <= root_index - 1:
            root.left = self.construct_optimal_BST(allNodes, allRoots, beg, root_index - 1)
        if root_index + 1 <= end:
            root.right = self.construct_optimal_BST(allNodes, allRoots, root_index + 1, end)
        return root

    def insert_optimal_BST(self, node_list: [int], node_weights: [[int]]):
        allNodes = [Node(key) for key in node_list]
        subtrees_cost, allRoots = self.calculate_optimal_BST(node_weights, len(allNodes))
        self.root = self.construct_optimal_BST(allNodes, allRoots, 0, len(allNodes) - 1)

    def calculate_optimal_BST(self, node_weights, n):
        subtrees_cost = [[0] * n for _ in range(n)]
        allRoots = [[0] * n for _ in range(n)]
        W = [[0] * n for _ in range(n)]

        for i in range(n):
            for j in range(i, n):
                W[i][j] = node_weights[i][j]
                if i != j:
                    W[i][j] += W[i][j - 1] if j > 0 else 0

        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                subtrees_cost[i][j] = float('inf')
                for r in range(i, j + 1):
                    cost = (subtrees_cost[i][r - 1] if r > i else 0) + \
                           (subtrees_cost[r + 1][j] if r < j else 0) + \
                           W[i][j]
                    if cost < subtrees_cost[i][j]:
                        subtrees_cost[i][j] = cost
                        allRoots[i][j] = r

        return subtrees_cost, allRoots

    def construct_optimal_BST(self, allNodes, allRoots, beg, end):
        if beg > end:
            return None
        root_index = allRoots[beg][end]
        root = allNodes[root_index]
        root.left = self.construct_optimal_BST(allNodes, allRoots, beg, root_index - 1)
        root.right = self.construct_optimal_BST(allNodes, allRoots, root_index + 1, end)
        return root