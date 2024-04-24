from topology.Node import Node
class SplayNet:

    def __init__(self):
        self.root = None  # root of the BST
        self.serviceCost = 0
        self.rotationCost = 0


    def setInsertionOver(self):
        self.insertionOver = True

    def getRotationCost(self):
        return self.rotationCost

    def getServiceCost(self):
        return self.serviceCost

    def increaseRotationCost(self, cost):
        self.rotationCost += cost

    def increaseServingCost(self, cost):
        if cost < 0:
            raise Exception("distance < 0")
        if cost == 0:
            raise Exception("distance = 0")
        self.serviceCost += cost

    def insertBalancedBST(self, nodeList):
        nodeList.sort()
        k = len(nodeList) // 2
        self.root = Node(nodeList[k])
        self.root.left = self.insertionIteration(nodeList[:k], self.root)
        self.root.right = self.insertionIteration(nodeList[k + 1:], self.root)

    def insertionIteration(self, nodeList, parent):
        if len(nodeList) == 0:
            return None
        k = len(nodeList) // 2
        newNode = Node(nodeList[k])
        newNode.parent = parent
        newNode.left = self.insertionIteration(nodeList[:k], newNode)
        newNode.right = self.insertionIteration(nodeList[k + 1:], newNode)
        return newNode

    def assignLastParents(self, node, left, right):
        if node:
            node.lastLeftParent = left
            node.lastRightParent = right
            self.assignLastParents(node.left, left, node.key)
            self.assignLastParents(node.right, node.key, right)

    def commute(self, u, v):
        uNode = u
        vNode = v
        common_ancestor = self.findLCA(uNode, vNode)
        if common_ancestor.key == vNode:
            uNode, vNode = vNode, uNode
        parent_CA = common_ancestor.parent or common_ancestor
        if parent_CA.key > common_ancestor.key:
            newLCA = self.splay_new(common_ancestor, uNode)
        elif parent_CA.key < common_ancestor.key:
            newLCA = self.splay_new(common_ancestor, uNode)
        else:
            newLCA = self.splay_new(self.root, uNode)
        if uNode == vNode:
            raise Exception("gleiche Knoten kommunizieren")
        if uNode > vNode:
            self.splay_new(newLCA.left, vNode)
        if uNode < vNode:
            self.splay_new(newLCA.right, vNode)
        distance = self.calculate_distance(uNode, vNode)
        self.increaseServingCost(distance)

    def findLCA(self, u, v):
        node = self.root
        while node and ((u > node.key and v > node.key) or (u < node.key and v < node.key)):
            if u > node.key:
                node = node.right
            else:
                node = node.left
        assert node is not None
        return node

    def splay_new(self, h, key):
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

    def splay_up(self, h, k):
        if h == k:
            return k
        elif k.parent == h:
            if h.left == k:
                self.rotateRight(h)
            elif h.right == k:
                self.rotateLeft(h)
            else:
                raise Exception("H should be parent, but not has k as child")
            return k
        else:
            found = h == k.parent.parent
            if k.parent.parent.right == k.parent:
                if k.parent.right == k:
                    self.rotateLeft(k.parent.parent)
                elif k.parent.left == k:
                    self.rotateRight(k.parent)
                else:
                    raise Exception("k.p has not k as child")
                self.rotateLeft(k.parent)
            elif k.parent.parent.left == k.parent:
                if k.parent.right == k:
                    self.rotateLeft(k.parent)
                elif k.parent.left == k:
                    self.rotateRight(k.parent.parent)
                else:
                    raise Exception("k.p has not k as child")
                self.rotateRight(k.parent)
            else:
                raise Exception("k.p.p has not k.p as child")
            if found:
                return k
        return self.splay_up(h, k)

    def rotateRight(self, h):
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
        self.increaseRotationCost(cost)
        x.lastRightParent = h.lastRightParent
        h.lastLeftParent = x.key
        return x

    def rotateLeft(self, h):
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
        self.increaseRotationCost(cost)
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
        else:
            print("Node is None")
