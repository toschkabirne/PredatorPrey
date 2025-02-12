import pygame
#import random
#import math
#import numpy as np
from module.Brain_Neural_Network import *
from module.settings import *

# Hilfsfunktionen
def wrap_position(pos, width, height):
    """Behandle Spielfeldränder: Wenn ein Objekt den Rand verlässt, erscheint es auf der gegenüberliegenden Seite."""
    x, y = pos
    x = x % width
    y = y % height
    return x, y

def distance(a, b):
    """Berechne die Entfernung zwischen zwei Punkten."""
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

class Predator:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.angle = random.uniform(0, 2 * math.pi)
        self.energy = PRED_ENERGY
        self.brain = NeuralNetwork(NUMBER_SIGHTS_PREDATOR, 2, PRED_INIT_MUT)
        self.eatenPrey = 0

    def get_inputs(self, preys):
        inputs = [0] * NUMBER_SIGHTS_PREDATOR
        start_angle = self.angle - math.radians(30)
        indexListe = list(range(NUMBER_SIGHTS_PREDATOR))
        for prey in preys:
            if distance((self.x, self.y), (prey.x, prey.y)) < SIGHT_RANGE_PREDATOR: #überprüfe jeden prey, wenn nah, überprüfe 
                for i in indexListe: #indexListe #...überprüfe jede sichtlinie, und gebe als input rein.
                    sight_angle = start_angle + i * math.radians(6)
                    if abs(math.atan2(prey.y - self.y, prey.x - self.x) - sight_angle) < math.radians(3):
                        inputs[i] = 1
                        indexListe.remove(i)
        return inputs

    def move(self, inputs): #scales good
        self.energy -= PRED_DEFAULT_DECAY
        outputs = self.brain.forward_vectorized(inputs, self.energy/PRED_ENERGY)
        self.angle += outputs[1] * math.pi
        self.x += outputs[0] * PREDATOR_SPEED * math.cos(self.angle)
        self.y += outputs[0] * PREDATOR_SPEED * math.sin(self.angle)
        self.x, self.y = wrap_position((self.x, self.y), SCREEN_WIDTH, SCREEN_HEIGHT)
        self.energy -= outputs[0] * PRED_MOVING_DECAY

    def hunt(self, preys):
        for prey in preys:
            if distance((self.x, self.y), (prey.x, prey.y)) < 10:
                self.energy = min(PRED_ENERGY, self.energy + PREDATOR_ENERGY_GAIN)
                self.eatenPrey +=1
                preys.remove(prey)

    def reproduce(self):
        if self.eatenPrey > 3:
            self.eatenPrey = 0
            offspring = Predator(self.x + random.randint(-1,1), self.y + random.randint(-1,1))
            offspring.brain = self.brain
            for _ in range(random.randint(2,6)):
                offspring.brain.mutate()
            return offspring
        return None

    def draw_sight(self, screen):
        """Zeichne Sichtlinien des Predators."""
        start_angle = self.angle - math.radians(30)
        end_angle = self.angle + math.radians(30)
        for i in range(NUMBER_SIGHTS_PREDATOR):
            sight_angle = start_angle + i * (end_angle - start_angle) / (NUMBER_SIGHTS_PREDATOR-1)
            end_x = self.x + SIGHT_RANGE_PREDATOR * math.cos(sight_angle)
            end_y = self.y + SIGHT_RANGE_PREDATOR * math.sin(sight_angle)
            pygame.draw.line(screen, (255, 255, 0), (self.x, self.y), (end_x, end_y), 1)

    def draw(self, screen):
        pygame.draw.circle(screen, (255, 0, 0), (int(self.x), int(self.y)), 10)
        self.draw_sight(screen)

class Prey:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.angle = random.uniform(0, 2 * math.pi)
        self.energy = PREY_ENERGY
        self.rest_time = 0
        self.brain = NeuralNetwork(NUMBER_SIGHTS_PREY, 2, PREY_INIT_MUT)

    def get_inputs(self, spatialpredators, predators):
        inputs = [0] * NUMBER_SIGHTS_PREY
        indexListe = list(range(NUMBER_SIGHTS_PREY))
        for predator in spatialpredators:
            if distance((self.x, self.y), (predator.x, predator.y)) < 10:
                predator.energy = min(PRED_ENERGY, predator.energy + PREDATOR_ENERGY_GAIN)
                predator.eatenPrey +=1
                new_predator = predator.reproduce()
                if new_predator:
                   predators.append(new_predator)
                return None
            if distance((self.x, self.y), (predator.x, predator.y)) < SIGHT_RANGE_PREY:
                poss_remov = []
                for i in indexListe:
                    sight_angle = self.angle + i * math.radians(360/NUMBER_SIGHTS_PREY)
                    if abs(math.atan2(predator.y - self.y, predator.x - self.x) - sight_angle) < math.radians(360/(2*NUMBER_SIGHTS_PREY)):
                        inputs[i] = 1
                        poss_remov.append(i)
                for i in poss_remov:
                    indexListe.remove(i)
                if indexListe: 
                    continue
                else:
                    break
                        
        return inputs
        

    def move(self, inputs):
        outputs = self.brain.forward_vectorized(inputs, self.energy/PREY_ENERGY)
        if outputs[0] == 0: #if ouputs[0] <= 0.001
            self.energy = min(PREY_ENERGY , self.energy+PREY_REST_ENERGY_GAIN)
            #self.rest_time += 1
            return
        if self.energy < 0:
            return
        self.angle += outputs[1] * math.pi
        self.x += outputs[0] * PREY_SPEED * math.cos(self.angle)
        self.y += outputs[0] * PREY_SPEED * math.sin(self.angle)
        self.x, self.y = wrap_position((self.x, self.y), SCREEN_WIDTH, SCREEN_HEIGHT)
        self.energy -= outputs[0] * PREY_MOVING_DECAY

    #def rest(self):   

    def reproduce(self):
        self.rest_time += 1
        if self.rest_time >= PREY_REPRODUCATION_RATE*FRAMES_PER_SECOND:
            self.rest_time = 0
            offspring = Prey(random.randint(int(self.x-50), int(self.x+50)), random.randint(int(self.y-50), int(self.y+50)))
            offspring.brain = self.brain
            for _ in range(random.randint(2,6)):
                offspring.brain.mutate()
            return offspring
        return None

    def draw_sight(self, screen):
        """Zeichne Sichtlinien des Preys."""
        for i in range(NUMBER_SIGHTS_PREY):
            sight_angle = self.angle + i * math.radians(360/NUMBER_SIGHTS_PREY) 
            end_x = self.x + SIGHT_RANGE_PREY * math.cos(sight_angle)
            end_y = self.y + SIGHT_RANGE_PREY * math.sin(sight_angle)
            pygame.draw.line(screen, (0, 255, 255), (self.x, self.y), (end_x, end_y), 1)

    def draw(self, screen):
        pygame.draw.circle(screen, (0, 255, 0), (int(self.x), int(self.y)), 7)
        self.draw_sight(screen)
        

def hihi():
    nn = NeuralNetwork(num_inputs=10, num_outputs=2, mutate = 10)
    #for _ in range(30): 
     #   nn.mutate()
    """hidden1 = nn.add_neuron('hidden')
    hidden2 = nn.add_neuron('hidden')
    nn.add_neuron()
    nn.add_neuron()
    nn.add_neuron()
    # Verbindungen hinzufügen
    nn.add_connection(nn.input_neurons[0].id, hidden1.id, 0.5)
    nn.add_connection(hidden1.id, nn.output_neurons[0].id, -0.7)
    for _ in range(20):
        nn.add_random_connection()
    """
    
    # Netzwerk visualisieren
    
    liste = [[random.randint(0,1) for _ in range(nn.num_inputs)] for _ in range(10)]
    for element in liste:
        print(nn.forward(element), end= " ")
    print("")
    plot_neural_network(nn)

# Beispielnetzwerk erstellen
if __name__ == "__main__":
    hihi()