import sys
from goat_routing.algorithm.map.map_generator_sim import SimulatedMap
from goat_routing.algorithm.map.map_interface import Mode, Node
from goat_routing.algorithm.route_map import RouteMap
from goat_routing.algorithm.router import Router


if __name__ == "__main__":
    time_priority = 1
    map = SimulatedMap(graph_res=32, seed=200964)
    route_map = RouteMap(map=map)
    start = Node(0, 0, Mode.DRIVING)
    end = Node(14, 14, Mode.DRIVING)
    router = Router(route_map=route_map)  # Replace with actual start, goal, and graph
    path = router.astar(start, end)
    if path is None:
        print("No route found.")
    else:
        map.visualize_with_path(path)
        map._map.show()
    print(f"Route with time priority '{time_priority}' created successfully.")
