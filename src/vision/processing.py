import cv2
import numpy as np
import logging

class ImageProcessor:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.logger.info("ImageProcessor initialized with configuration parameters")
        
        # Get parameters from config
        self.roi_vertices = np.array(config['roi_vertices'], np.int32)
        self.hough_params = config['hough_transform']
        
    def preprocess(self, frame):
        """Preprocess image for lane detection: grayscale conversion, Gaussian blur, and Canny edge detection"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Canny edge detection
        edges = cv2.Canny(blur, 50, 150)
        self.logger.debug("Image preprocessing completed")
        
        return edges
    
    def apply_roi(self, edges):
        """Apply Region of Interest mask to focus on relevant road area"""
        height, width = edges.shape
        mask = np.zeros_like(edges)
        
        # Fill polygon with white (255) to create ROI mask
        cv2.fillPoly(mask, [self.roi_vertices], 255)
        
        # Apply mask to edge-detected image
        roi_edges = cv2.bitwise_and(edges, mask)
        self.logger.debug("Region of Interest applied")
        
        return roi_edges
    
    def detect_lanes(self, roi_edges):
        """Detect lane lines using Hough Transform"""
        lines = cv2.HoughLinesP(
            roi_edges,
            rho=self.hough_params['rho'],
            theta=self.hough_params['theta'],
            threshold=self.hough_params['threshold'],
            minLineLength=self.hough_params['min_line_length'],
            maxLineGap=self.hough_params['max_line_gap']
        )
        
        if lines is not None:
            self.logger.debug(f"Detected {len(lines)} lane line segments")
        else:
            self.logger.warning("No lane lines detected")
        
        return lines
    
    def calculate_lane_center(self, lines, image_width):
        """Calculate lane center based on detected lane lines"""
        if lines is None:
            self.logger.warning("Cannot calculate lane center - no lines detected")
            return image_width // 2
        
        left_lines = []
        right_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Skip vertical lines to avoid division by zero
            if x2 == x1:
                continue
            
            # Calculate line slope
            slope = (y2 - y1) / (x2 - x1)
            
            # Classify lines as left (negative slope) or right (positive slope)
            if slope < -0.5:
                left_lines.append((x1, y1, x2, y2))
            elif slope > 0.5:
                right_lines.append((x1, y1, x2, y2))
        
        # Calculate average x-coordinate for left and right lines
        left_x = np.mean([(x1 + x2) / 2 for x1, y1, x2, y2 in left_lines]) if left_lines else 0
        right_x = np.mean([(x1 + x2) / 2 for x1, y1, x2, y2 in right_lines]) if right_lines else image_width
        
        # Calculate lane center
        lane_center = (left_x + right_x) / 2
        self.logger.debug(f"Calculated lane center: {lane_center:.2f}")
        
        return lane_center
    
    def process(self, frame):
        """Complete image processing pipeline: preprocessing -> ROI -> lane detection -> center calculation"""
        edges = self.preprocess(frame)
        roi_edges = self.apply_roi(edges)
        lines = self.detect_lanes(roi_edges)
        lane_center = self.calculate_lane_center(lines, frame.shape[1])
        
        return edges, lines, lane_center