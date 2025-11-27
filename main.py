from module.Animals import *
#import pygame
#import random
#import time


class SpatialHash:
    def __init__(self, cell_size):
        self.cell_size = cell_size
        self.cells = {}

    def _hash(self, x, y):
        return (x // self.cell_size, y // self.cell_size)

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
                self.cells[old_cell].remove(obj)
                if not self.cells[old_cell]:  # Entferne leere Zellen
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

# Hauptspielfunktion
def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Predator and Prey Simulation")
    clock = pygame.time.Clock()

    predators = [Predator(random.randint(0, SCREEN_WIDTH), random.randint(0, SCREEN_HEIGHT)) for _ in range(PRED_INIT_NUMB)]
    preys = [Prey(random.randint(0, SCREEN_WIDTH), random.randint(0, SCREEN_HEIGHT)) for _ in range(PREY_INIT_NUMB)]
    spatialPreds = SpatialHash( SCREEN_WIDTH // SIGHT_RANGE_PREDATOR)
    spatialPreys = SpatialHash(SCREEN_WIDTH // SIGHT_RANGE_PREY)
    
    running = True
    while running:
        start_time = time.time()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill((0, 0, 0))
        for predator in predators: 
            spatialPreds.insert(predator)
        for prey in preys: 
            spatialPreys.insert(prey)
            
            # Bewege und zeichne Predator
        for predator in predators:
            #posssiblePrey = spatialPreys.query(predator.x, predator.y)
            inputs = predator.get_inputs(spatialPreys.query(predator.x, predator.y))
            predator.move(inputs)
            if predator.energy < 0:
                predators.remove(predator)
            #predator.hunt(posssiblePrey)
            #new_predators = predator.reproduce()
            #if new_predators:
             #   predators.extend(new_predators)
            predator.draw(screen)
            
        # Bewege und zeichne Prey
        for prey in preys:
            #prey.rest()
            inputs = prey.get_inputs(spatialPreds.query(prey.x, prey.y), predators) #newborn predator will not be drawn immediatly, maybe not bad
            if inputs:
                prey.move(inputs)
            else: 
                preys.remove(prey)
                continue
            if len(preys) < 900:
                new_prey = prey.reproduce()
                if new_prey:
                    preys.append(new_prey)
            prey.draw(screen)
            
        spatialPreds.cells = {}
        spatialPreys.cells = {}
        pygame.display.flip()
        end_time = time.time()  # Zeit am Ende der Iteration messen
        iteration_time = end_time - start_time  # Dauer berechnen
        # (f"Dauer der Iteration: {iteration_time:.6f} Sekunden and {len(preys)} and pred: {len(predators)}")
        clock.tick(FRAMES_PER_SECOND)

    pygame.quit()

if __name__ == "__main__":
    main()
