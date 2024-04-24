class Node:
    def __init__(self, key: int):
        self.key = key
        self.parent = None
        self.left = None
        self.right = None
        self.lastLeftParent = 0
        self.lastRightParent = float('inf')

    def __repr__(self):
        return f"Node {self.key}"

    def get_key(self):
        return self.key

    def get_parent(self):
        return self.parent

    def get_left(self):
        return self.left

    def get_right(self):
        return self.right

    # Setter eventuell überflüssig
    def set_parent(self, parent):
        self.parent = parent

    def set_left(self, left):
        self.left = left

    def set_right(self, right):
        self.right = right
