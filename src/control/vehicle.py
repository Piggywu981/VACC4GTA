import time
import logging
from pynput.keyboard import Controller

class VehicleController:
    def __init__(self, config):
        self.keyboard = Controller()
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.steering_threshold = config['steering']['deviation_threshold']
        self.steering_duration = config['steering']['steering_duration']
        self.logger.info("Vehicle controller initialized with steering parameters")
        
    def accelerate(self):
        """Simulate pressing the acceleration key (W)"""
        self.logger.debug("Pressing 'W' key to accelerate")
        self.keyboard.press('w')
        time.sleep(0.1)
        self.keyboard.release('w')
        self.logger.debug("Released 'W' key after acceleration")
        
    def turn_left(self):
        """Simulate pressing the left turn key (A)"""
        self.logger.debug(f"Pressing 'A' key to turn left for {self.steering_duration} seconds")
        self.keyboard.press('a')
        time.sleep(self.steering_duration)
        self.keyboard.release('a')
        self.logger.debug(f"Released 'A' key after left turn")
        
    def turn_right(self):
        """Simulate pressing the right turn key (D)"""
        self.logger.debug(f"Pressing 'D' key to turn right for {self.steering_duration} seconds")
        self.keyboard.press('d')
        time.sleep(self.steering_duration)
        self.keyboard.release('d')
        self.logger.debug(f"Released 'D' key after right turn")
        
    def apply_controls(self, lane_center, image_width):
        """Apply steering controls based on lane center deviation"""
        # Calculate deviation from center
        image_center = image_width // 2
        deviation = lane_center - image_center
        
        # Apply steering based on deviation
        if deviation > self.steering_threshold:
            self.turn_left()
        elif deviation < -self.steering_threshold:
            self.turn_right()
        else:
            self.logger.debug(f"Maintaining straight course. Deviation: {deviation:.2f}")
        
        # Always maintain acceleration
        self.accelerate()
        
    def stop(self):
        """Stop vehicle movement"""
        self.logger.debug("Pressing 'S' key to stop vehicle")
        self.keyboard.press('s')
        time.sleep(0.5)
        self.keyboard.release('s')
        self.logger.debug("Released 'S' key after stopping")
        self.logger.info("Vehicle control stopped")