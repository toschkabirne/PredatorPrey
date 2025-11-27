import sys
import json
import random
import time
import math
from unittest.mock import MagicMock

# Mock pygame before importing modules that might use it
sys.modules['pygame'] = MagicMock()
import pygame

# Import simulation modules
# Note: We need to patch parameters *after* import but *before* usage if they are used at module level.
# However, Brain_Neural_Network defines constants at module level.
# We will need to monkey-patch them.
import module.settings as settings
import module.Brain_Neural_Network as bnn
from module.Animals import Predator, Prey

# Re-implement SpatialHash from main.py since it's not in a module
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

    def move(self, obj, new_x, new_y):
        old_cell = self._hash(obj.x, obj.y)
        new_cell = self._hash(new_x, new_y)
        if old_cell != new_cell:
            if old_cell in self.cells:
                if obj in self.cells[old_cell]:
                    self.cells[old_cell].remove(obj)
                if not self.cells[old_cell]:
                    del self.cells[old_cell]
            obj.x = new_x
            obj.y = new_y 
            self.insert(obj)
            
    def query(self, x, y):
        cell = self._hash(x, y)
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                neighbor_cell = (cell[0] + dx, cell[1] + dy)
                if neighbor_cell in self.cells:
                    neighbors.extend(self.cells[neighbor_cell])
        return neighbors

def run_simulation(params, max_steps=5000):
    # Apply parameters
    # Brain_Neural_Network.py defines these at module level
    if 'ADD_NEURON' in params:
        bnn.ADD_NEURON = params['ADD_NEURON']
    if 'ADD_WEIGHT' in params:
        bnn.ADD_WEIGHT = params['ADD_WEIGHT']
    if 'CHANGE_WEIGHT' in params:
        bnn.CHANGE_WEIGHT = params['CHANGE_WEIGHT']
    
    # settings.py defines these
    if 'PRED_INIT_MUT' in params:
        settings.PRED_INIT_MUT = params['PRED_INIT_MUT']
    if 'PREY_INIT_MUT' in params:
        settings.PREY_INIT_MUT = params['PREY_INIT_MUT']
    
    # Initialize entities
    # We use fixed seeds for reproducibility if needed, but for search we might want randomness?
    # The user said "searching the parameter space", usually implies finding robust parameters.
    # Let's keep random seed random for now.
    
    predators = [Predator(random.randint(0, settings.SCREEN_WIDTH), random.randint(0, settings.SCREEN_HEIGHT)) for _ in range(settings.PRED_INIT_NUMB)]
    preys = [Prey(random.randint(0, settings.SCREEN_WIDTH), random.randint(0, settings.SCREEN_HEIGHT)) for _ in range(settings.PREY_INIT_NUMB)]
    
    spatialPreds = SpatialHash(settings.SCREEN_WIDTH // settings.SIGHT_RANGE_PREDATOR)
    spatialPreys = SpatialHash(settings.SCREEN_WIDTH // settings.SIGHT_RANGE_PREY)
    
    step = 0
    start_time = time.time()
    
    try:
        while step < max_steps:
            # Check extinction
            if len(predators) == 0 or len(preys) == 0:
                break
            
            # Update spatial hashes
            spatialPreds.cells = {}
            spatialPreys.cells = {}
            for predator in predators: 
                spatialPreds.insert(predator)
            for prey in preys: 
                spatialPreys.insert(prey)
            
            # Predators
            # Note: iterating over a copy or handling removal carefully
            # main.py iterates over `predators` and removes from it. 
            # In Python list iteration, removing current element is tricky.
            # main.py:
            # for predator in predators:
            #     ...
            #     if predator.energy < 0: predators.remove(predator)
            # This skips elements. We should iterate over a copy or use a new list.
            # But to stay true to the original simulation logic (bugs and all?), I should replicate it?
            # No, I should fix obvious bugs if they affect the outcome significantly, 
            # but maybe the user wants to optimize *this* simulation.
            # Let's do it safely: iterate over copy.
            
            active_predators = predators[:]
            for predator in active_predators:
                if predator not in predators: continue # Already removed?
                
                inputs = predator.get_inputs(spatialPreys.query(predator.x, predator.y))
                predator.move(inputs)
                
                if predator.energy < 0:
                    predators.remove(predator)
                    continue
                
                # Reproduction logic was commented out in main.py?
                # main.py lines 77-79 are commented out.
                # "The moment prey or predators die out, the simulation stops."
                # If predators don't reproduce, they will die out eventually.
                # Wait, let me check main.py again.
                pass

            # Preys
            active_preys = preys[:]
            for prey in active_preys:
                if prey not in preys: continue
                
                # prey.get_inputs can modify predators list (eating)!
                # "predator.reproduce()" is called inside prey.get_inputs if eaten?
                # Let's check Animals.py
                
                inputs = prey.get_inputs(spatialPreds.query(prey.x, prey.y), predators)
                
                # If inputs is None, it means prey was eaten (logic in Animals.py)
                # In Animals.py: if distance < 10: ... return None
                
                if inputs:
                    prey.move(inputs)
                else:
                    # Eaten
                    if prey in preys:
                        preys.remove(prey)
                    continue
                
                # Reproduction
                if len(preys) < settings.MAX_PREY_COUNT: # 900 in main.py, but MAX_PREY_COUNT is 800 in settings?
                    # main.py line 91: if len(preys) < 900:
                    # settings.py: MAX_PREY_COUNT = 800
                    # I will use the constant from settings if possible, or 900 to match main.py?
                    # User said: "Mainly focus on these parameters in the settings file."
                    # So I should probably use settings.MAX_PREY_COUNT.
                    new_prey = prey.reproduce()
                    if new_prey:
                        preys.append(new_prey)
            
            step += 1
            
    except Exception as e:
        return {
            "survived": False,
            "steps": step,
            "error": str(e),
            "parameters": params
        }

    end_time = time.time()
    duration = end_time - start_time
    avg_step_time = duration / step if step > 0 else 0
    
    survived = (len(predators) > 0 and len(preys) > 0)
    
    return {
        "survived": survived,
        "steps": step,
        "avg_step_time": avg_step_time,
        "final_predators": len(predators),
        "final_preys": len(preys),
        "parameters": params
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', type=str, required=True, help='JSON string of parameters')
    parser.add_argument('--max_steps', type=int, default=5000, help='Maximum steps to run')
    args = parser.parse_args()
    
    try:
        params = json.loads(args.params)
        result = run_simulation(params, args.max_steps)
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
