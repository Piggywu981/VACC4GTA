import cv2
import numpy as np
import mss
import time
import logging
import os
from src.data.config import ConfigManager
from src.vision.processing import ImageProcessor
from src.control.vehicle import VehicleController

# Initialize logging directory
LOG_DIR = 'logs'
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# Main program class
class GTAAutoDrive:
    def __init__(self):
        # Load configuration
        self.config_manager = ConfigManager()
        config = self.config_manager.config
        
        # Configure logging
        log_level = getattr(logging, config['logging']['level'].upper(), logging.INFO)
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(LOG_DIR, 'gta_auto_drive.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.image_processor = ImageProcessor(config)
        self.vehicle_controller = VehicleController(config)
        self.monitor = config['monitor']
        self.frame_delay = config['general']['frame_delay']
        
        self.running = False
        self.logger.info("GTA Autonomous Driving System initialized successfully")
        
    def run(self):
        """Run the autonomous driving main loop"""
        self.running = True
        with mss.mss() as sct:
            self.logger.info("Autonomous driving program started. Press Ctrl+C to stop...")
            print("Autonomous driving program started. Press Ctrl+C to stop...")
            frame_count = 0
            
            try:
                while self.running:
                    frame_count += 1
                    # Capture screen frame
                    sct_img = sct.grab(self.monitor)
                    frame = np.array(sct_img)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                    
                    # Process image
                    edges, lines, lane_center, arrow_info = self.image_processor.process(frame)
                    
                    # Apply vehicle controls
                    self.vehicle_controller.apply_controls(lane_center, arrow_info, frame.shape[1])
                    
                    # Display processed image
                    cv2.imshow('Processed Frame', frame)
                    cv2.setWindowProperty('Processed Frame', cv2.WND_PROP_TOPMOST, 1)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.stop()
                        break
                    
                    # Control loop frequency
                    time.sleep(self.frame_delay)
                    
            except KeyboardInterrupt:
                self.logger.info("Program interrupted by user")
            except Exception as e:
                self.logger.error(f"Program error occurred: {str(e)}", exc_info=True)
            finally:
                self.stop()
                cv2.destroyAllWindows()
        
    def stop(self):
        """Stop the autonomous driving system"""
        if self.running:
            self.running = False
            self.vehicle_controller.stop()
            self.logger.info("Autonomous driving program stopped")
            print("Autonomous driving program stopped")

if __name__ == "__main__":
    try:
        auto_drive = GTAAutoDrive()
        auto_drive.run()
    except Exception as e:
        print(f"Program startup failed: {str(e)}")
        logging.error(f"Program startup failed: {str(e)}", exc_info=True)