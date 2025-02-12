import NeuronalNetwork as mm
import pygame
import random
import math
from module.settings import *

#PREDATOR
#Movement
#PREDATOR_SPEED = 25
SIGHT_RANGE_PREDATOR = 100
NUMBER_SIGHTS_PREDATOR = 6
PRED_TIME_MOVE_DIST_WIDTH = 10
PREDATOR_SPEED = SCREEN_WIDTH/(FRAMES_PER_SECOND*PRED_TIME_MOVE_DIST_WIDTH)

#Energy
PRED_ENERGY = 100
PREDATOR_ENERGY_GAIN = 40 #kann ich noch abhängig von der totalen populationsgröße machen, je weniger pray, desto weniger energy_gain
PREDATOR_LIFESPAN = 15
PRED_DEFAULT_DECAY = PRED_ENERGY / (PREDATOR_LIFESPAN*FRAMES_PER_SECOND)
PRED_MOVING_DECAY = PRED_ENERGY/(2*SCREEN_WIDTH) #pred_energy/(2*pred_speed*10*30)

#PREY
#Movement
PREY_SPEED = 0.8 * PREDATOR_SPEED  # Preys starten mit Geschwindigkeit 0
SIGHT_RANGE_PREY = 40
NUMBER_SIGHTS_PREY = 15
#Energy
PREY_ENERGY = 50
PREY_REPRODUCATION_RATE = 5
PREY_MOVING_DECAY = PREY_ENERGY/(0.8*SCREEN_WIDTH)
PREY_REST_ENERGY_GAIN = PREY_ENERGY/(4*FRAMES_PER_SECOND)

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
        self.brain = mm.NeuralNetwork(NUMBER_SIGHTS_PREDATOR, 2, PRED_INIT_MUT)
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
        outputs = self.brain.forward(inputs)
        self.angle = outputs[1] * (2 * math.pi)
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
        if self.eatenPrey > 2:
            self.eatenPrey = 0
            offsprings = []
            for _ in range(2):
                offspring = Predator(self.x, self.y)
                offspring.brain = self.brain
                for _ in range(random.randint(2,6)):
                    offspring.brain.mutate()
                offsprings.append(offspring)
            return offsprings
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
        self.brain = mm.NeuralNetwork(NUMBER_SIGHTS_PREY, 2, PREY_INIT_MUT)

    def get_inputs(self, predators):
        inputs = [0] * NUMBER_SIGHTS_PREY
        indexListe = list(range(NUMBER_SIGHTS_PREY))
        for predator in predators:
            if distance((self.x, self.y), (predator.x, predator.y)) < SIGHT_RANGE_PREY:
                for i in indexListe:
                    sight_angle = self.angle + i * math.radians(360/NUMBER_SIGHTS_PREY)
                    if abs(math.atan2(predator.y - self.y, predator.x - self.x) - sight_angle) < math.radians(360/(2*NUMBER_SIGHTS_PREY)):
                        inputs[i] = 1
                        indexListe.remove(i)
                        
        return inputs

    def move(self, inputs):
        outputs = self.brain.forward(inputs)
        if outputs[0] < 0.05: #if ouputs[0] <= 0.001
            self.energy = min(PREY_ENERGY , self.energy+PREY_REST_ENERGY_GAIN)
            #self.rest_time +=1
            return
        if self.energy < 0:
            return
        self.angle = outputs[1] * (2 * math.pi)
        self.x += outputs[0] * PREY_SPEED * math.cos(self.angle)
        self.y += outputs[0] * PREY_SPEED * math.sin(self.angle)
        self.x, self.y = wrap_position((self.x, self.y), SCREEN_WIDTH, SCREEN_HEIGHT)
        self.energy -= outputs[0] * PREY_MOVING_DECAY

    #def rest(self):   

    def reproduce(self):
        self.rest_time += 1
        if self.rest_time >= PREY_REPRODUCATION_RATE*FRAMES_PER_SECOND:
            self.rest_time = 0
            offspring = Prey(self.x, self.y)
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