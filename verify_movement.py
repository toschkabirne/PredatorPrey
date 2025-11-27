import pygame
import math
from module.Animals import Predator, Prey, PRED_ENERGY, PREY_ENERGY, PRED_MOVING_DECAY, PREY_MOVING_DECAY, PREY_REST_ENERGY_GAIN, NUMBER_SIGHTS_PREY, NUMBER_SIGHTS_PREDATOR

# Mock pygame to avoid window opening
pygame.init()
pygame.display.set_mode((1, 1))

def test_inputs():
    print("Testing Inputs...")
    pred = Predator(0, 0)
    pred.angle = 0
    
    # Test 1: Prey far away
    prey_far = Prey(100, 0) # Distance 100
    inputs_far = pred.get_inputs([prey_far])
    active_far = sum(inputs_far)
    print(f"Far Prey (100px): {active_far} inputs active")
    
    # Test 2: Prey close
    prey_close = Prey(20, 0) # Distance 20
    inputs_close = pred.get_inputs([prey_close])
    active_close = sum(inputs_close)
    print(f"Close Prey (20px): {active_close} inputs active")
    
    if active_close > active_far:
        print("PASS: Closer prey triggers more inputs.")
    else:
        print("FAIL: Closer prey did not trigger more inputs.")

def test_movement_physics():
    print("\nTesting Movement Physics...")
    prey = Prey(0, 0)
    prey.energy = 50
    initial_energy = prey.energy
    
    # Test 1: Resting (Low output)
    # We can't easily force NN output without mocking, but we can call move with mocked inputs if we mock the brain?
    # Actually, move() calls brain.forward_vectorized(). 
    # Let's mock the brain.
    
    class MockBrain:
        def forward_vectorized(self, inputs, energy_ratio):
            return [0.01, 0] # Low speed, 0 turn
            
    prey.brain = MockBrain()
    prey.move([0]*NUMBER_SIGHTS_PREY)
    
    if prey.energy > initial_energy:
        print(f"PASS: Resting gained energy. Old: {initial_energy}, New: {prey.energy}")
    else:
        print(f"FAIL: Resting did not gain energy. Old: {initial_energy}, New: {prey.energy}")
        
    # Test 2: Sprinting (High output)
    prey.energy = 50
    initial_energy = prey.energy
    
    class MockBrainFast:
        def forward_vectorized(self, inputs, energy_ratio):
            return [1.0, 0] # Max speed
            
    prey.brain = MockBrainFast()
    prey.move([0]*NUMBER_SIGHTS_PREY)
    
    expected_decay = (1.0**2) * PREY_MOVING_DECAY
    # Note: move() also calls wrap_position which doesn't affect energy.
    # And it might have other costs? No, just PREY_MOVING_DECAY * speed^2
    
    # Wait, in the code: self.energy -= (speed_factor ** 2) * PREY_MOVING_DECAY
    
    print(f"Sprinting energy: {prey.energy}, Expected change: -{expected_decay}")
    if prey.energy < initial_energy:
        print("PASS: Sprinting consumed energy.")
    else:
        print("FAIL: Sprinting did not consume energy.")

if __name__ == "__main__":
    test_inputs()
    test_movement_physics()
