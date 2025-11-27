import sys
import time
import random
import cProfile
import pstats

# Mock pygame BEFORE importing modules that use it
class MockPygame:
    def init(self): pass
    def quit(self): pass
    class display:
        @staticmethod
        def set_mode(*args): return None
        @staticmethod
        def set_caption(*args): pass
        @staticmethod
        def flip(): pass
    class event:
        @staticmethod
        def get(): return []
        class type:
            QUIT = 123
    class time:
        class Clock:
            def tick(self, *args): pass
    class draw:
        @staticmethod
        def circle(*args): pass
        @staticmethod
        def line(*args): pass
    QUIT = 123

sys.modules['pygame'] = MockPygame()

from module.Animals import Predator, Prey
from module.settings import *
import module.Animals
module.Animals.pygame = MockPygame()
import main
main.pygame = MockPygame()

class SpatialHash:
    def __init__(self, cell_size):
        self.cell_size = cell_size
        self.cells = {}

    def _hash(self, x, y):
        return (int(x // self.cell_size), int(y // self.cell_size))

    def insert(self, obj):
        cell = self._hash(obj.x, obj.y)
        if cell not in self.cells:
            self.cells[cell] = []
        self.cells[cell].append(obj)

    def query(self, x, y):
        cell = self._hash(x, y)
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                neighbor_cell = (cell[0] + dx, cell[1] + dy)
                if neighbor_cell in self.cells:
                    neighbors.extend(self.cells[neighbor_cell])
        return neighbors

def run_simulation(steps=100):
    # Setup
    predators = [Predator(random.randint(0, SCREEN_WIDTH), random.randint(0, SCREEN_HEIGHT)) for _ in range(PRED_INIT_NUMB)]
    preys = [Prey(random.randint(0, SCREEN_WIDTH), random.randint(0, SCREEN_HEIGHT)) for _ in range(PREY_INIT_NUMB)]
    spatialPreds = SpatialHash(SCREEN_WIDTH // SIGHT_RANGE_PREDATOR)
    spatialPreys = SpatialHash(SCREEN_WIDTH // SIGHT_RANGE_PREY)

    start_time = time.time()
    
    for _ in range(steps):
        # Clear spatial hash
        spatialPreds.cells = {}
        spatialPreys.cells = {}
        
        # Insert into spatial hash
        for predator in predators: 
            spatialPreds.insert(predator)
        for prey in preys: 
            spatialPreys.insert(prey)
            
        # Move Predators
        for predator in predators:
            inputs = predator.get_inputs(spatialPreys.query(predator.x, predator.y))
            predator.move(inputs)
            if predator.energy < 0:
                predators.remove(predator)
            # Reproduction logic omitted for pure movement profiling, or keep if critical
            
        # Move Prey
        for prey in preys:
            inputs = prey.get_inputs(spatialPreds.query(prey.x, prey.y), predators)
            if inputs:
                prey.move(inputs)
            else: 
                preys.remove(prey)
                continue
            
            # Reproduction logic
            if len(preys) < 900:
                new_prey = prey.reproduce()
                if new_prey:
                    preys.append(new_prey)

    end_time = time.time()
    print(f"Simulation finished in {end_time - start_time:.4f} seconds for {steps} steps.")
    print(f"Final counts: Predators={len(predators)}, Prey={len(preys)}")

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    run_simulation(steps=200)
    profiler.disable()
    
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats(20)
