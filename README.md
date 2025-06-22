# GTA5 Autonomous Driving Program (VACC4GTA)

A vision-based autonomous driving assistance system for GTA5 that implements real-time screen capture, image processing, lane detection, and vehicle control through modular design.

## Features
- **Modular Architecture**: Clear separation of vision processing, vehicle control, and configuration management
- **Real-time Image Processing**: Grayscale conversion, edge detection, and lane line recognition
- **Intelligent Vehicle Control**: Adaptive steering system based on lane center deviation
- **Configurable Parameters**: Adjust all key algorithm parameters through JSON file
- **Detailed Logging**: Simultaneous output to file and console for easy debugging

## Project Structure
```
VACC4GTA/
├── .venv/              # Python virtual environment
├── logs/               # Log files directory
├── src/
│   ├── vision/
│   │   └── processing.py    # Image processing module: lane detection and image analysis
│   ├── control/
│   │   └── vehicle.py       # Vehicle control module: keyboard input and driving logic
│   └── data/
│       └── config.py        # Configuration management module: load and parse configuration files
├── main.py             # Main program entry: system initialization and main loop
├── config.json         # Configuration file: adjust algorithm parameters and system settings
├── requirements.txt    # Dependencies list: manage project dependencies
└── README.md           # Project documentation
```

## System Requirements
- Python 3.8+
- Windows operating system (supports DirectX/OpenGL window capture)
- GTA5 (recommended to run in windowed mode, 1920x1080 resolution)
- Administrator privileges (to ensure proper keyboard input simulation)

## Installation Steps
1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd VACC4GTA
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Activate virtual environment on Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Configuration Instructions
Modify the `config.json` file to adjust system parameters:
- **monitor**: Screen capture area (top, left, width, height)
- **roi_vertices**: Region of interest vertices (lane detection area)
- **hough_transform**: Hough transform parameters (line detection sensitivity)
- **steering**: Steering control parameters (deviation threshold, steering duration)
- **logging**: Log level (DEBUG/INFO/WARNING/ERROR)
- **general**: General parameters (frame delay, debug mode)

## Usage Guide
1. Ensure GTA5 is running in windowed mode with resolution matching the configuration in `config.json`
2. Run the program
   ```bash
   python main.py
   ```
3. The program will automatically start capturing the screen and controlling the vehicle
4. **Operation Shortcuts**:
   - `q`: Close the image display window
   - `Ctrl+C`: Stop program execution

## Core Modules Explanation
### 1. Image Processing Module (`src/vision/processing.py`)
- **Image Preprocessing**: Grayscale conversion, Gaussian blur, Canny edge detection
- **Region of Interest**: Focus on road area to reduce interference
- **Lane Detection**: Hough transform line detection algorithm
- **Lane Center Calculation**: Compute center position based on detected lane lines

### 2. Vehicle Control Module (`src/control/vehicle.py`)
- **Keyboard Input Simulation**: Use pynput library to simulate driving operations
- **Steering Control**: Calculate steering amount based on lane center deviation
- **Speed Control**: Basic forward movement (extensible to adaptive cruise control)

### 3. Configuration Management Module (`src/data/config.py`)
- **Configuration Loading**: Parse JSON configuration file
- **Parameter Validation**: Ensure configuration values are valid
- **Error Handling**: Handle missing configuration files or format errors

## Notes
- Please carefully configure the screen capture area to match the game window before first run
- Lane detection performance is affected by lighting and weather conditions
- Recommended to test on empty roads to avoid complex traffic environments
- Log files are saved in `logs/gta_auto_drive.log`
- The program requires the game window to be active to control the vehicle

## Troubleshooting
- **Dependency installation failed**: Ensure virtual environment is activated and try updating pip: `python -m pip install --upgrade pip`
- **Screen capture issues**: Verify game window resolution matches configuration
- **Control unresponsive**: Ensure game window has focus, try running program as administrator
- **Inaccurate lane detection**: Adjust ROI area or Hough transform parameters in `config.json`

## Future Improvements
- Integrate deep learning models to improve lane detection accuracy
- Add traffic light and traffic sign recognition
- Implement adaptive cruise control
- Develop collision warning and avoidance system
- Add user graphical interface configuration tool