import numpy as np
from goat_routing.algorithm.map.map_interface import Mode, Node, MapBase, Terrain
from enum import Enum


class ModeCost(Enum):
    DRIVING = 1.0
    ROLLING = 0.0
    FLYING = 2.0
    SWIMMING = 1.0


class ModeSpeed(Enum):
    DRIVING = 1.0
    ROLLING = 5.0
    FLYING = 5.0
    SWIMMING = 0.1


class RouteMap:
    """
    Interface with map data to provide possible paths, nodes and costs.
    """

    map: MapBase
    planar_directions = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
    ]  # Up, Down, Left, Right

    def __init__(self, map: MapBase):
        self.map = map

    def heuristic_cost_estimate(self, current: Node, goal: Node):
        # Turn into dijkstra by returning zero
        return 0

    def distance_between(self, n1: Node, n2: Node) -> float:
        # Transition between modes
        if n1.x == n2.x and n1.y == n2.y and n1.mode != n2.mode:
            return 0.0

        if n1.mode != n2.mode:
            raise ValueError("Nodes must be in the same mode to calculate distance.")

        # It's a grid so always same distance between neighbors
        return 1 / ModeSpeed[n1.mode.name].value * ModeCost[n1.mode.name].value

    def neighbors(self, node: Node):
        neighbors = []

        # Append mode change on the spot
        for mode in Mode:
            if mode != node.mode:
                neighbors.append(Node(node.x, node.y, mode))

        # Append planar movements in current mode
        for dx, dy in self.planar_directions:
            nx, ny = node.x + dx, node.y + dy

            if 0 > nx or nx >= self.map.get_num_nodes() or 0 > ny or ny >= self.map.get_num_nodes():
                continue

            next_node = Node(nx, ny, node.mode)
            if self.reachable(node, next_node):
                neighbors.append(next_node)

        return neighbors

    def reachable(self, start: Node, end: Node):
        terrain_type = self.map.get_terrain_type(start, end)
        mode = start.mode
        if terrain_type == Terrain.CLIFF:
            return True if mode == Mode.FLYING else False
        if terrain_type == Terrain.WATER:
            return True if mode == Mode.SWIMMING or mode == Mode.FLYING else False
        if terrain_type == Terrain.SLOPE:
            if mode == Mode.DRIVING:
                slope = self.map.get_slope(start, end)
                return slope < 0.3
            if mode == Mode.ROLLING:
                slope = self.map.get_slope(start, end)
                return slope < -0.2
        if terrain_type == Terrain.FLAT:
            return True if mode == Mode.DRIVING or mode == Mode.FLYING else False
        return False
