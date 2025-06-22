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
        self.max_steering_angle = config['steering']['max_steering_angle']
        self.logger.debug("Vehicle controller initialized with steering parameters")
        
    def accelerate(self):
        """Simulate pressing the acceleration key (W)"""
        self.logger.debug("Pressing 'W' key to accelerate")
        self.keyboard.press('w')
        time.sleep(1)
        self.keyboard.release('w')
        self.logger.debug("Released 'W' key after acceleration")
        
    def turn_left(self, duration_factor=1.0):
        """Simulate pressing the left turn key (A) with variable duration"""
        actual_duration = self.steering_duration * duration_factor
        self.logger.debug(f"Pressing 'A' key to turn left for {actual_duration:.2f} seconds")
        self.keyboard.press('a')
        time.sleep(actual_duration)
        self.keyboard.release('a')
        self.logger.debug(f"Released 'A' key after left turn")
        
    def turn_right(self, duration_factor=1.0):
        """Simulate pressing the right turn key (D) with variable duration"""
        actual_duration = self.steering_duration * duration_factor
        self.logger.debug(f"Pressing 'D' key to turn right for {actual_duration:.2f} seconds")
        self.keyboard.press('d')
        time.sleep(actual_duration)
        self.keyboard.release('d')
        self.logger.debug(f"Released 'D' key after right turn")
        
    def apply_controls(self, lane_center, arrow_info, image_width):
        """Apply steering controls based on lane center and arrow direction"""
        # Priority 1: Follow arrow direction if detected
        if arrow_info:
            cx, cy, direction = arrow_info
            self.logger.info(f"Following arrow direction: {direction}")
            
            # Apply direction-specific steering
            # Calculate relative position of arrow to lane center
            arrow_deviation = cx - lane_center
            max_deviation = image_width // 2  # Maximum possible deviation
            
            # Calculate steering angle based on deviation ratio
            steering_ratio = min(abs(arrow_deviation) / max_deviation, 1.0)
            steering_angle = steering_ratio * self.max_steering_angle
            duration_factor = steering_angle / (self.max_steering_angle / 2.0)  # Map angle to duration factor
            
            # Determine direction based on arrow position relative to lane
            if arrow_deviation < -10:  # Arrow is on the left side of lane center
                self.turn_right(duration_factor)
            elif arrow_deviation > 10:  # Arrow is on the right side of lane center
                self.turn_left(duration_factor)
            # If arrow is near center, maintain straight course

            return
        
        # Priority 2: Fallback to lane center following
        image_center = image_width // 2
        deviation = lane_center - image_center
        duration_factor = min(abs(deviation) / self.steering_threshold, 2.0)
        
        if deviation > self.steering_threshold:
            self.turn_left(duration_factor)
        elif deviation < -self.steering_threshold:
            self.turn_right(duration_factor)
        else:
            self.logger.info(f"Maintaining straight course. Deviation: {deviation:.2f}")
        
        # Always maintain acceleration
        #self.accelerate()
        
    def stop(self):
        """Stop vehicle movement"""
        self.logger.debug("Pressing 'S' key to stop vehicle")
        self.keyboard.release('w')  # Stop acceleration
        self.keyboard.press('s')
        time.sleep(0.5)
        self.keyboard.release('s')
        self.logger.info("Released 'S' key after stopping")
        self.logger.info("Vehicle control stopped")