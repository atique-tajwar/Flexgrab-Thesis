import serial
import numpy as np
import cv2

# --- Configuration ---
PORT = 'COM5'  # Change this to match your ESP32's port (e.g., '/dev/ttyUSB0' on Mac/Linux)
BAUD = 115200

def main():
    try:
        ser = serial.Serial(PORT, BAUD, timeout=1)
        print(f"Connected to {PORT}. Select the video window and press 'q' to quit.")
    except Exception as e:
        print(f"Failed to connect to {PORT}: {e}")
        return

    sync_marker = b'\xff\xff\xaa\x55'

    while True:
        # Ask the ESP32 for a frame
        ser.write(b'F')
        
        # Wait until we see the 4-byte sync sequence
        ser.read_until(sync_marker)
        
        # Read exactly 1,296 bytes (the 36x36 pixel grid)
        raw_data = ser.read(1296)
        
        if len(raw_data) == 1296:
            # Convert raw bytes into an 8-bit unsigned integer array
            img_array = np.frombuffer(raw_data, dtype=np.uint8)
            
            # Reshape it into a 2D 36x36 grid
            img = img_array.reshape((36, 36))
            
            # 36x36 is too small to see on a monitor. 
            # We scale it up by 10x using nearest-neighbor interpolation to keep it sharp.
            img_scaled = cv2.resize(img, (360, 360), interpolation=cv2.INTER_NEAREST)
            
            # Display the image
            cv2.imshow("PAW3395 Surface View", img_scaled)
            
            # Wait 10ms for the frame to render, break if 'q' is pressed
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    ser.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()