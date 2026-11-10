import serial
import cv2
import numpy as np

# --- Configuration ---
PORT = 'COM11'  # Make sure this matches your ESP32!
BAUD = 115200
WINDOW_SIZE = 800
SENSITIVITY = 0.5  # Lower this if the line flies off the screen too fast

def main():
    try:
        ser = serial.Serial(PORT, BAUD, timeout=0.1)
        print(f"Connected to {PORT}.")
        print("Live Tracking Started! Press 'q' to quit, 'c' to clear the canvas.")
    except Exception as e:
        print(f"Failed to connect: {e}")
        return

    # Create a blank black canvas and start in the dead center
    canvas = np.zeros((WINDOW_SIZE, WINDOW_SIZE, 3), dtype=np.uint8)
    curr_x, curr_y = WINDOW_SIZE // 2, WINDOW_SIZE // 2

    while True:
        try:
            # Read all available lines in the serial buffer
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8').strip()
                
                # Check if the line matches our "dx,dy" format
                if "," in line:
                    dx_str, dy_str = line.split(',')
                    dx, dy = int(dx_str), int(dy_str)

                    # Calculate new position
                    new_x = int(curr_x + (dx * SENSITIVITY))
                    # Note: We subtract 'dy' because screen Y-coordinates go top-to-bottom, 
                    # but standard math/sensor Y-coordinates go bottom-to-top.
                    new_y = int(curr_y - (dy * SENSITIVITY)) 

                    # Prevent the line from going off the edge of the screen
                    new_x = max(0, min(WINDOW_SIZE - 1, new_x))
                    new_y = max(0, min(WINDOW_SIZE - 1, new_y))

                    # Draw a line from the old position to the new position
                    cv2.line(canvas, (curr_x, curr_y), (new_x, new_y), (0, 255, 0), 2)
                    
                    # Update current position
                    curr_x, curr_y = new_x, new_y

            # Show the updated canvas
            cv2.imshow("PAW3395 Live Tracker", canvas)
            
            # Keyboard controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                # Reset canvas to black and move position back to center
                canvas = np.zeros((WINDOW_SIZE, WINDOW_SIZE, 3), dtype=np.uint8)
                curr_x, curr_y = WINDOW_SIZE // 2, WINDOW_SIZE // 2

        except ValueError:
            # Ignore garbled serial data if it happens during startup
            pass 
        except KeyboardInterrupt:
            break

    ser.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()