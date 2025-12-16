import numpy as np
from scipy import ndimage
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List
import random
from goat_routing.algorithm.map.map_interface import MapBase, Node, Terrain, Mode
from world_map_generator.generation import FractalGenerator
from world_map_generator.map import Map
from world_map_generator.utils import Bounding
from world_map_generator.rendering import save_height_map_as_image
from world_map_generator.default_values import DEFAULT_DIAMOND_SQUARE_GRID_MAX_VALUE


COLORS = {
    Terrain.FLAT: "go",
    Terrain.SLOPE: "yo",
    Terrain.CLIFF: "ko", # black
    Terrain.WATER: "bo",
}


MODE_COLORS = {
    Mode.ROLLING: "c",  # cyan
    Mode.DRIVING: "g",  # green
    Mode.SWIMMING: "b",  # blue
    Mode.FLYING: "r",  # red
}


class NaturalHeightMap:
    """
    A class to generate natural-looking N x N height maps with mountains,
    rivers, slopes, and flat areas.

    DISCLAIMER: Partially AI Generated Code
    """

    def __init__(self, size: int = 512, seed: Optional[int] = None):
        """
        Initialize the height map generator.

        Args:
            size: Size of the height map (N x N)
            seed: Random seed for reproducibility
        """
        self.size = size
        self.chunk_width = 64
        self.num_chunks = self.size // self.chunk_width
        self.chunked_map: Map = None
        self.height_map: np.ndarray = None
        self._water_threshold = 0.35  # Elevation below which is considered water

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

    def _add_river(self, terrain: np.ndarray) -> np.ndarray:
        """
        Add sine wave rivers across the terrain.
        """
        result = terrain.copy()

        # Determine river parameters
        amplitude = np.random.uniform(self.size // 10, self.size // 6)
        frequency = np.random.uniform(0.01, 0.03)
        phase = np.random.uniform(0, 2 * np.pi)

        # Main river starts near center
        start_y = np.random.randint(self.size // 4, 3 * self.size // 4)

        # Create sine wave path
        river_width = np.random.randint(5, 10)  # Width in pixels

        # Generate river path
        for x in range(self.size):
            # Calculate y position using sine wave
            y_offset = amplitude * np.sin(frequency * x + phase)
            river_y = int(start_y + y_offset)

            # Ensure y is within bounds
            river_y = max(0, min(self.size - 1, river_y))

            # Carve river channel
            for dy in range(-river_width, river_width + 1):
                for dx in range(-1, 2):  # Small x variation for width
                    px = min(max(0, x + dx), self.size - 1)
                    py = min(max(0, river_y + dy), self.size - 1)

                    # Calculate distance from center
                    dist = abs(dy)

                    if dist <= river_width:
                        # Carve deeper in center, shallower at edges
                        depth = 0.2 * (1 - dist / river_width)
                        result[px, py] = max(0.01, result[px, py] - depth)

        # Smooth the river edges
        result = ndimage.gaussian_filter(result, sigma=0.7)

        # Ensure rivers don't go below sea level
        result = np.maximum(result, 0.0)

        return result

    def _normalize_terrain(self, terrain: np.ndarray) -> np.ndarray:
        """
        Normalize terrain to 0-1 range and apply final adjustments.
        """
        # Normalize
        terrain_normalized = (terrain - terrain.min()) / (terrain.max() - terrain.min())

        return terrain_normalized

    def generate(self) -> np.ndarray:
        """
        Generate a complete natural height map.

        Returns:
            N x N height map array normalized to 0-1
        """

        self.chunked_map = Map(chunk_width=self.chunk_width)
        generator = FractalGenerator(self.chunked_map.seed, self.chunk_width, self.chunk_width)
        for i in range(self.num_chunks):
            for j in range(self.num_chunks):
                self.chunked_map.set_chunk(generator.generate_chunk(i, j))

        terrain = np.zeros((self.size, self.size))
        for i in range(self.num_chunks):
            for j in range(self.num_chunks):
                chunk = self.chunked_map.get_chunk(i, j)
                if chunk is not None:
                    for x in range(self.chunk_width):
                        for y in range(self.chunk_width):
                            global_x = i * self.chunk_width + x
                            global_y = j * self.chunk_width + y
                            terrain[global_x, global_y] = chunk.tiles[x][y]

        terrain = self._normalize_terrain(terrain)

        # terrain = self._normalize_terrain(terrain)
        
        terrain = np.clip(terrain, self._water_threshold, 1.0)

        terrain = self._add_river(terrain)

        self.height_map  = terrain

        # temporarily set all heights to 1 for testing
        # self.height_map = np.ones_like(self.height_map) * 0.5

        return self.height_map

    def visualize(self) -> None:
        """
        Visualize the generated height map.
        """
        if self.height_map is None:
            raise ValueError("Height map not generated. Call generate() first.")

        fig = plt.figure(figsize=(15, 5))

        # 2D height map
        ax1 = fig.add_subplot(121)
        im = ax1.imshow(self.height_map, cmap="terrain", origin="lower")
        ax1.set_title("2D Height Map")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        plt.colorbar(im, ax=ax1, label="Elevation")

        # Contour plot
        ax2 = fig.add_subplot(122)
        # ax2 = fig.add_subplot(111)
        contour = ax2.contour(self.height_map, levels=10, colors="black", linewidths=0.5)
        ax2.imshow(self.height_map, cmap="terrain", alpha=0.7, origin="lower")
        ax2.set_title("Contour Lines")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")

    def show(self, save_path: Optional[str] = None) -> None:
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        plt.show()

    def save_heightmap(self, filename: str, format: str = "npy") -> None:
        """
        Save the height map to a file.

        Args:
            filename: Output filename
            format: File format ('npy', 'txt', or 'png')
        """
        if self.height_map is None:
            raise ValueError("Height map not generated. Call generate() first.")

        save_height_map_as_image(self.chunked_map, filename + "_gen", max_color_value=1.5 * DEFAULT_DIAMOND_SQUARE_GRID_MAX_VALUE)

        filename = f"{filename}.{format}"

        if format == "npy":
            np.save(filename, self.height_map)
        elif format == "txt":
            np.savetxt(filename, self.height_map)
        elif format == "png":
            # Scale to 0-255 for PNG
            img_data = (self.height_map * 255).astype(np.uint8)
            plt.imsave(filename, img_data, cmap="gray")
        else:
            raise ValueError(f"Unsupported format: {format}")


class SimulatedMap(MapBase):

    size: int = 512
    max_nodes: int

    def __init__(self, graph_res: int = 16, seed: Optional[int] = None):
        MapBase.__init__(self, graph_res=graph_res)

        self._map = NaturalHeightMap(size=self.size, seed=seed)
        self._map.generate()

        # check if graph res is valid
        if self.size % graph_res != 0:
            raise ValueError("Graph resolution must evenly divide the map size.")

        self.max_nodes: int = self.size // graph_res - 1

    def is_within_bounds(self, node: Node) -> bool:
        """
        Check if a node is within the map bounds.
        """
        return 0 <= node.x < self.max_nodes and 0 <= node.y < self.max_nodes

    def get_slope(self, start: Node, end: Node) -> float:
        """
        Get average slope between two nodes.
        """
        if self._map.height_map is None:
            raise ValueError("Height map not generated. Call generate() first.")

        x0, y0 = self.graph_res * (start.x + 1), self.graph_res * (start.y + 1)
        x1, y1 = self.graph_res * (end.x + 1), self.graph_res * (end.y + 1)

        start_height = self._map.height_map[x0, y0]
        end_height = self._map.height_map[x1, y1]
        height_diff = end_height - start_height
        distance = np.hypot(x1 - x0, y1 - y0)
        slope = height_diff / distance

        return slope

    def get_node_terrain_type(self, node_x: int, node_y: int) -> Terrain:
        """
        Get terrain type at a specific node.
        """
        if self._map.height_map is None:
            raise ValueError("Height map not generated. Call generate() first.")

        x, y = self.graph_res * (node_x + 1), self.graph_res * (node_y + 1)

        # sample in circular area around node
        sample_radius = self.graph_res // 2
        xs = np.arange(max(0, x - sample_radius), min(self.size, x + sample_radius))
        ys = np.arange(max(0, y - sample_radius), min(self.size, y + sample_radius))
        xx, yy = np.meshgrid(xs, ys)
        distances = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
        mask = distances <= sample_radius
        heights = self._map.height_map[xx[mask], yy[mask]]  

        min_height = np.min(heights)
        max_height = np.max(heights)
        height_diff = max_height - min_height
        if min_height < self._map._water_threshold:
            return Terrain.WATER
        if height_diff < 0.1:
            return Terrain.FLAT
        elif height_diff < 0.25:
            return Terrain.SLOPE
        else:
            return Terrain.CLIFF

    def get_terrain_type(self, start: Node, end: Node) -> Terrain:
        """
        Get terrain type between two nodes by sampling the terrain in a straight line.
        """
        if self._map.height_map is None:
            raise ValueError("Height map not generated. Call generate() first.")
        
        start_type = self.get_node_terrain_type(start.x, start.y)
        end_type = self.get_node_terrain_type(end.x, end.y)

        if start_type == Terrain.WATER or end_type == Terrain.WATER:
            return Terrain.WATER
        
        if start_type == Terrain.CLIFF or end_type == Terrain.CLIFF:
            return Terrain.CLIFF
        
        if start_type == Terrain.SLOPE or end_type == Terrain.SLOPE:
            return Terrain.SLOPE
        
        if start_type == Terrain.FLAT or end_type == Terrain.FLAT:
            return Terrain.FLAT

    def visualize_with_nodes(self):
        self._map.visualize()

        for x in range(self.max_nodes):
            for y in range(self.max_nodes):
                node_type = self.get_node_terrain_type(x, y)
                color = COLORS[node_type]
                plt.plot(self.graph_res * (y + 1), self.graph_res * (x + 1), color, markersize=2)

    def visualize_with_path(self, path):
        self.visualize_with_nodes()
        for node in path:
            plt.plot(self.graph_res * (node.y + 1), self.graph_res * (node.x + 1), "r.", markersize=6)


    def get_num_nodes(self) -> int:
        return self.max_nodes

# Example usage
if __name__ == "__main__":
    # Create a 256x256 height map
    map = SimulatedMap(graph_res=16, seed=200964)

    start = Node(2, 2, mode=Terrain.FLAT)
    end = Node(2, 3, mode=Terrain.FLAT)
    slope = map.get_slope(start, end)
    terrain_type = map.get_terrain_type(start, end)

    # Visualize the result
    # map.visualize()
    map.visualize_with_nodes()
    map._map.show()

    # Save the height map
    map._map.save_heightmap("terrain_heightmap", format="png")
