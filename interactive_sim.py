import pygame
import sys
import random
from module.settings import *
from module.Animals import Prey
from module.ControlledAnimals import ControlledPredator
import math

def main():
    # Prompt for mutation count
    try:
        mutations_input = input("How many mutations for the prey? (Default: 12): ")
        if mutations_input.strip() == "":
            mutations = 12
        else:
            mutations = int(mutations_input)
            
    except ValueError:
        print("Invalid input. Using default 12.")
        mutations = 12

    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Interactive Predator-Prey")
    clock = pygame.time.Clock()

    # Spawn entities
    # Prey in middle
    prey = Prey(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
    
    # Predator to the right
    predator = ControlledPredator(SCREEN_WIDTH // 2 + 200, SCREEN_HEIGHT // 2)
    predator.angle = math.pi # Face left towards prey
    
    # Apply mutations to prey brain
    # Prey init already creates a brain with PREY_INIT_MUT (12). 
    # If user wants more/different, we can mutate more.
    
    from module.Brain_Neural_Network import NeuralNetwork
    prey.brain = NeuralNetwork(NUMBER_SIGHTS_PREY, 2, mutations)

    running = True
    while running:
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Input handling for Predator
        keys = pygame.key.get_pressed()
        predator.move(keys)
        
        # Let's just pass [predator] as spatialpredators for simplicity since there is only 1.
        inputs = prey.get_inputs([predator], [predator])
        
        # Check if prey was eaten (get_inputs returns None if eaten)
        if inputs is None:
            print("Prey eaten! Spawning new prey.")
            prey = Prey(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
            prey.brain = NeuralNetwork(NUMBER_SIGHTS_PREY, 2, mutations)
        else:
            prey.move(inputs)

        # Drawing
        screen.fill((0, 0, 0)) # Black background
        
        predator.draw(screen)
        prey.draw(screen)
        
        # Draw Neural Network of Prey
        # Scale 0.8 smaller. Original was 300x400. Let's say 240x320.
        draw_neural_network(screen, prey.brain, 10, 10, 240, 320)

        pygame.display.flip()
        clock.tick(FRAMES_PER_SECOND)

    pygame.quit()
    sys.exit()

def draw_neural_network(screen, nn, x, y, width, height):
    if not hasattr(nn, 'last_activations') or not hasattr(nn, 'last_inputs'):
        return

    # Draw Frame and Background
    pygame.draw.rect(screen, (255, 255, 255), (x, y, width, height)) # White background
    pygame.draw.rect(screen, (0, 0, 0), (x, y, width, height), 2) # Black frame

    inputs = nn.last_inputs
    activations = nn.last_activations # Hidden + Output (size: neuron_number - (num_inputs+1))
    
    num_inputs = nn.num_inputs
    num_outputs = nn.num_outputs
    total_neurons = nn.neuron_number
    num_hidden = total_neurons - (num_inputs + 1 + num_outputs)
    
    # Define positions
    input_pos = []
    output_pos = []
    hidden_pos = []
    
    # Inputs (Left)
    for i in range(num_inputs):
        pos_x = x + 20
        pos_y = y + (height / (num_inputs + 1)) * (i + 1)
        input_pos.append((pos_x, pos_y))
        
    # Bias (Left, below inputs)
    bias_pos = (x + 20, y + (height / (num_inputs + 1)) * (num_inputs + 1))
    
    # Outputs (Right)
    for i in range(num_outputs):
        pos_x = x + width - 20
        pos_y = y + (height / (num_outputs + 1)) * (i + 1)
        output_pos.append((pos_x, pos_y))
        
    # Hidden (Middle)
    for i in range(num_hidden):
        offset_x = (width / 2) + (i % 3 - 1) * 30 
        pos_x = x + offset_x
        pos_y = y + (height / (num_hidden + 1)) * (i + 1)
        hidden_pos.append((pos_x, pos_y))
        
    node_positions = {}
    for i in range(num_inputs):
        node_positions[i] = input_pos[i]
    node_positions[num_inputs] = bias_pos
    
    in_bi = num_inputs + 1
    for i in range(num_outputs):
        node_positions[in_bi + i] = output_pos[i]
        
    for i in range(num_hidden):
        node_positions[in_bi + num_outputs + i] = hidden_pos[i]
        
    # Helper to get activation value
    def get_activation(id):
        if id < num_inputs:
            return inputs[id]
        elif id == num_inputs:
            return 1.0 
        else:
            return activations[id - in_bi]

    # Draw Edges
    # Input Matrix
    rows, cols = nn.Input_Matrix.shape
    for r in range(rows): # Target: in_bi + r
        for c in range(cols): # Source: c
            weight = nn.Input_Matrix[r, c]
            if weight == 0: continue # No connection
            
            source_id = c
            target_id = in_bi + r
            
            source_val = get_activation(source_id)
            val = source_val * weight
            
            start = node_positions[source_id]
            end = node_positions[target_id]
            
            # Draw all edges
            # If triggered (source_val > 0 and weight != 0), color it.
            # Else draw gray.
            
            if source_val > 0:
                intensity = min(255, int(abs(val) * 255))
                # if intensity < 20: continue # Draw faint ones too if requested "all edges"
                color = (0, intensity, 0) if val > 0 else (intensity, 0, 0)
                width_line = 2 if intensity > 50 else 1
                pygame.draw.line(screen, color, start, end, width_line)
            else:
                # Inactive edge (source not firing)
                pygame.draw.line(screen, (200, 200, 200), start, end, 1)

    # Hidden Matrix
    rows, cols = nn.Hidden_Matrix.shape
    for r in range(rows):
        for c in range(cols):
            weight = nn.Hidden_Matrix[r, c]
            if weight == 0: continue
            
            source_id = in_bi + c
            target_id = in_bi + r
            
            source_val = get_activation(source_id)
            val = source_val * weight
            
            start = node_positions[source_id]
            end = node_positions[target_id]
            
            if source_val > 0:
                intensity = min(255, int(abs(val) * 255))
                color = (0, intensity, 0) if val > 0 else (intensity, 0, 0)
                width_line = 2 if intensity > 50 else 1
                pygame.draw.line(screen, color, start, end, width_line)
            else:
                pygame.draw.line(screen, (200, 200, 200), start, end, 1)

    # Draw Nodes
    for id, pos in node_positions.items():
        val = get_activation(id)
        
        c_val = int(max(-1, min(1, val)) * 255)
        if c_val > 0:
            color = (0, c_val, 0)
        else:
            color = (abs(c_val), 0, 0)
            
        # If value is very low, maybe draw gray?
        if abs(c_val) < 20:
            color = (150, 150, 150)
            
        pygame.draw.circle(screen, color, (int(pos[0]), int(pos[1])), 5)
        pygame.draw.circle(screen, (0, 0, 0), (int(pos[0]), int(pos[1])), 5, 1) # Outline

if __name__ == "__main__":
    main()
