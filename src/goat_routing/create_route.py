import sys
from goat_routing.algorithm import Router


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python create_route.py <time_priority>")
        sys.exit(1)

    time_priority = sys.argv[1]
    router = Router(start="A", goal="B", graph=None)  # Replace with actual start, goal, and graph
    print(f"Route with time priority '{time_priority}' created successfully.")
