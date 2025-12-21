import pygame
import math
from module.settings import *
from module.Animals import wrap_position

class ControlledPredator:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.angle = 0
        self.energy = PRED_ENERGY
        self.eatenPrey = 0
        # No brain needed

    def move(self, keys):
        # keys is a dictionary or list of pressed keys
        
        speed_factor = 0
        rotation = 0
        
        if keys[pygame.K_UP]:
            speed_factor = 1.0
        if keys[pygame.K_DOWN]:
            speed_factor = -1.0
            
        if keys[pygame.K_LEFT]:
            rotation = -0.1 # Adjust rotation speed as needed
        if keys[pygame.K_RIGHT]:
            rotation = 0.1
            
        self.angle += rotation * math.pi
        
        # Update position
        self.x += speed_factor * PREDATOR_SPEED * math.cos(self.angle)
        self.y += speed_factor * PREDATOR_SPEED * math.sin(self.angle)
        
        self.x, self.y = wrap_position((self.x, self.y), SCREEN_WIDTH, SCREEN_HEIGHT)
        
        # Simple energy decay (optional, but keeps it consistent with simulation)
        # self.energy -= ... 

    def draw(self, screen):
        # Draw predator as a red circle, maybe with a line indicating direction
        pygame.draw.circle(screen, (255, 0, 0), (int(self.x), int(self.y)), 10)
        
        # Draw direction indicator
        end_x = self.x + 20 * math.cos(self.angle)
        end_y = self.y + 20 * math.sin(self.angle)
        pygame.draw.line(screen, (200, 0, 0), (self.x, self.y), (end_x, end_y), 2)

    def reproduce(self):
        # Controlled predator doesn't reproduce in this mode, or just returns None
        return None
