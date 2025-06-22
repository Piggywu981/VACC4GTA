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
        # Apply Gaussian blur to reduce noise
        blur = cv2.GaussianBlur(frame, (5, 5), 0)
        
        # Convert to HSV color space for color isolation
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        
        # Define color ranges in HSV
        # Purple channel
        lower_purple = np.array([125, 50, 50])
        upper_purple = np.array([150, 255, 255])
        purple_mask = cv2.inRange(hsv, lower_purple, upper_purple)
        
        # Black channel (low value)
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 50])
        black_mask = cv2.inRange(hsv, lower_black, upper_black)
        
        # Combine masks and set as arrow mask
        self.arrow_mask = cv2.bitwise_or(purple_mask, black_mask)
        
        # Apply Canny edge detection on combined mask
        edges = cv2.Canny(self.arrow_mask, 50, 150)
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
    
    def detect_arrow_direction(self):
        """Detect arrow direction using contour analysis and shape matching"""
        contours, _ = cv2.findContours(self.arrow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Find largest contour (assumed to be the arrow)
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 100:
                # Get bounding rectangle and aspect ratio
                x, y, w, h = cv2.boundingRect(largest_contour)
                aspect_ratio = float(w) / h
                
                # Calculate contour moments for centroid
                M = cv2.moments(largest_contour)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    
                    # Simple direction classification based on contour shape
                    if aspect_ratio > 1.2:
                        direction = 'straight'
                    else:
                        # Check contour orientation using extreme points
                        leftmost = tuple(largest_contour[largest_contour[:, :, 0].argmin()][0])
                        rightmost = tuple(largest_contour[largest_contour[:, :, 0].argmax()][0])
                        topmost = tuple(largest_contour[largest_contour[:, :, 1].argmin()][0])
                        bottommost = tuple(largest_contour[largest_contour[:, :, 1].argmax()][0])
                        
                        # Determine arrow direction based on point positions
                        if (rightmost[0] - leftmost[0]) > (bottommost[1] - topmost[1]):
                            direction = 'straight'
                        # Only detect left, right and straight directions relative to purple route
                        if leftmost[0] < cx - w/4:
                            direction = 'left'
                        elif rightmost[0] > cx + w/4:
                            direction = 'right'
                        else:
                            direction = 'straight'
                    
                    self.logger.debug(f"Detected arrow at ({cx}, {cy}) with direction: {direction}")
                    return (cx, cy, direction)
        self.logger.warning("No arrow detected")
        return None

    def process(self, frame):
        """Complete image processing pipeline: preprocessing -> ROI -> lane detection -> center calculation"""
        edges = self.preprocess(frame)
        roi_edges = self.apply_roi(edges)
        lines = self.detect_lanes(roi_edges)
        lane_center = self.calculate_lane_center(lines, frame.shape[1])
        arrow_info = self.detect_arrow_direction()
        
        return edges, lines, lane_center, arrow_info