from avltree import AvlTree

class AvlTreeIris(AvlTree):
    
    def insert(self, key, value):
        self.__setitem__(key, value)

    def find_closest(self, composite_index):
        if self._AvlTree__root_key is None:
            return None

        closest_key = self._AvlTree__root_key
        min_diff = abs(composite_index - closest_key)

        current_key = self._AvlTree__root_key
        nodes = self._AvlTree__nodes

        while current_key is not None:
            current_node = nodes[current_key]
            current_diff = abs(composite_index - current_key)

            if current_diff < min_diff:
                min_diff = current_diff
                closest_key = current_key

            if composite_index < current_key:
                current_key = current_node.lesser_child_key
            elif composite_index > current_key:
                current_key = current_node.greater_child_key
            else:
                return current_key
        return closest_key

        
    def height(self):
        root_key = self._AvlTree__root_key
        if root_key is None:
            return -1

        nodes = self._AvlTree__nodes
        root_node = nodes.get(root_key)
        if root_node is None:
            return -1

        return root_node.height + 1

    def size(self):
        return self.__len__()