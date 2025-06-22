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
        self.keyboard.press('w')
        time.sleep(0.1)
        self.keyboard.release('w')
        
    def turn_left(self):
        """Simulate pressing the left turn key (A)"""
        self.keyboard.press('a')
        time.sleep(self.steering_duration)
        self.keyboard.release('a')
        self.logger.debug(f"Turned left for {self.steering_duration} seconds")
        
    def turn_right(self):
        """Simulate pressing the right turn key (D)"""
        self.keyboard.press('d')
        time.sleep(self.steering_duration)
        self.keyboard.release('d')
        self.logger.debug(f"Turned right for {self.steering_duration} seconds")
        
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
        # In case of emergency stop, we could add brake logic here
        self.logger.info("Vehicle control stopped")
        self.keyboard.press('s')
        time.sleep(0.5)
        self.keyboard.release('s')
        self.logger.info("Vehicle stopped")