import abc
from enum import Enum
from typing import Iterable


class Mode(Enum):
    """Ranked in energy usage."""

    ROLLING = 0
    DRIVING = 1
    SWIMMING = 2
    FLYING = 3


class Terrain(Enum):
    """Types of terrain."""

    FLAT = 0
    SLOPE = 1
    CLIFF = 2
    WATER = 3


class Node:
    """
    A node in the map grid.
    """

    x: int
    y: int
    mode: Mode

    def __init__(self, x: int, y: int, mode: Mode):
        self.x = x
        self.y = y
        self.mode = mode

    # return tuple representation
    def to_tuple(self):
        return (self.x, self.y, self.mode)

    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y, self.mode))


class MapBase(abc.ABC):
    """
    Abstract base class for map interfaces.
    """

    graph_res: int

    def __init__ (self, graph_res: int):
        self.graph_res = graph_res

    @abc.abstractmethod
    def get_slope(self, start: Node, end: Node):
        """
        Get the average slop between two nodes by sampling the terrain in a straight line.
        """
        pass

    @abc.abstractmethod
    def get_terrain_type(self, start: Node, end: Node):
        """
        Get terrain type between two nodes by sampling the terrain in a straight line.
        """
        pass

    @abc.abstractmethod
    def get_node_terrain_type(self, node: Node):
        """
        Get terrain type at a specific node.
        """
        pass

    @abc.abstractmethod
    def get_num_nodes(self) -> int:
        """
        Get the number of nodes along one dimension of the map.
        """
        pass

    @abc.abstractmethod
    def visualize_with_path(self, path: Iterable[Node]):
        """
        Visualize the map with a given path overlayed.
        """
        pass
