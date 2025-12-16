from astar import AStar
from goat_routing.algorithm.route_map import Node, RouteMap

class Router(AStar):
    def __init__(self, route_map: RouteMap):
        self.route_map: RouteMap =  route_map
    
    def heuristic_cost_estimate(self, current: Node, goal: Node):
        return self.route_map.heuristic_cost_estimate(current, goal)
    
    def distance_between(self, n1: Node, n2: Node):
        return self.route_map.distance_between(n1, n2)
    
    def neighbors(self, node: Node):
        return self.route_map.neighbors(node)