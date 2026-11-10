"""
FSR Data Logger
Reads ESP32 serial output continuously.
Press SPACEBAR once: starts logging incoming readings (Time, Voltage,
Resistance) into memory, up to SAMPLES_PER_SESSION (350).
Logging stops automatically at 350 samples (or earlier if SPACEBAR is pressed
again), then asks for a name and saves the CSV.
Press ESC: quits cleanly (discards any in-progress unsaved session).

Requires: pip install pyserial
(Uses built-in msvcrt, Windows only.)
"""

import serial
import csv
import re
import os
import time
import threading
import msvcrt
from datetime import datetime

# ---------------- CONFIG ----------------
SERIAL_PORT = "COM11"     # your ESP32 port
BAUD_RATE = 115200
DEBOUNCE_SECONDS = 0.4     # ignore repeated key events within this window
SAMPLES_PER_SESSION = 350  # stop automatically after this many readings
# -----------------------------------------

LINE_PATTERN = re.compile(
    r"Voltage:\s*([\d.]+)\s*V.*Resistance:\s*([\d.]+)\s*ohm"
)

latest_data = {"voltage": None, "resistance": None}
lock = threading.Lock()
stop_flag = False

logging_active = threading.Event()   # set = currently logging
paused_print = threading.Event()     # set = reader thread should not print (typing in progress)
session_ready = threading.Event()    # set = sample limit reached, ready to save
buffer = []
buffer_lock = threading.Lock()


def sanitize_filename(name):
    name = name.strip()
    invalid = '<>:"/\\|?*'
    for ch in invalid:
        name = name.replace(ch, "")
    return name if name else "unnamed"


def flush_keyboard_buffer():
    """Discard any queued keystrokes (e.g. from key-repeat / bounce)."""
    while msvcrt.kbhit():
        msvcrt.getch()


def read_line_raw(prompt):
    """
    Manually read a line of text using msvcrt, staying entirely in raw
    keypress mode. Avoids mixing with input(), which caused conflicts
    with the msvcrt-based keypress polling used elsewhere in this script.
    Supports basic typing + backspace, ends on Enter.
    """
    print(prompt, end="", flush=True)
    chars = []
    while True:
        ch = msvcrt.getch()

        if ch in (b"\r", b"\n"):  # Enter
            print()  # move to next line
            return "".join(chars).strip()

        elif ch == b"\x08":  # Backspace
            if chars:
                chars.pop()
                print("\b \b", end="", flush=True)  # erase last char visually

        elif ch == b"\x1b":  # ESC while typing -> cancel entry
            print()
            return ""

        elif ch in (b"\x00", b"\xe0"):
            # Special/function key prefix (arrows, F-keys) -> consume and ignore the next byte
            msvcrt.getch()

        else:
            try:
                decoded = ch.decode("utf-8")
            except UnicodeDecodeError:
                continue
            if decoded.isprintable():
                chars.append(decoded)
                print(decoded, end="", flush=True)


def serial_reader(ser):
    global stop_flag
    while not stop_flag:
        try:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if not line:
                continue

            match = LINE_PATTERN.search(line)
            if match:
                voltage = float(match.group(1))
                resistance = float(match.group(2))
                with lock:
                    latest_data["voltage"] = voltage
                    latest_data["resistance"] = resistance

                if paused_print.is_set():
                    continue  # don't print while user is typing a name

                if logging_active.is_set():
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    with buffer_lock:
                        if len(buffer) < SAMPLES_PER_SESSION:
                            buffer.append((timestamp, voltage, resistance))
                        count = len(buffer)

                    print(
                        f"\rLogging ({count}/{SAMPLES_PER_SESSION}) -> "
                        f"Voltage: {voltage:.3f} V | Resistance: {resistance:.1f} ohm   ",
                        end="",
                    )

                    if count >= SAMPLES_PER_SESSION:
                        logging_active.clear()
                        session_ready.set()
                        print(f"\nReached {SAMPLES_PER_SESSION} samples. Logging stopped.")
                else:
                    print(f"\rLive -> Voltage: {voltage:.3f} V | Resistance: {resistance:.1f} ohm   ", end="")
        except Exception as e:
            print(f"\nSerial read error: {e}")


def save_buffer_to_csv(name):
    filename = sanitize_filename(name) + ".csv"
    file_exists = os.path.isfile(filename)

    with buffer_lock:
        rows = list(buffer)
        buffer.clear()

    with open(filename, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Time", "Voltage (V)", "Resistance (ohm)"])
        writer.writerows(rows)

    return filename, len(rows)


def prompt_and_save_session():
    """Ask for a filename and save the buffer, or discard if cancelled."""
    with buffer_lock:
        row_count = len(buffer)

    print(f"\n{row_count} readings collected.")

    if row_count == 0:
        print("No data collected, nothing to save.\n")
        return

    paused_print.set()
    flush_keyboard_buffer()
    name = read_line_raw("Enter name to save this session as (CSV filename): ")
    paused_print.clear()

    if name:
        filename, saved_count = save_buffer_to_csv(name)
        print(f"Saved {saved_count} rows to {filename}\n")
    else:
        with buffer_lock:
            buffer.clear()
        print("No name entered, session discarded.\n")


def main():
    global stop_flag

    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    except Exception as e:
        print(f"Could not open serial port {SERIAL_PORT}: {e}")
        return

    time.sleep(2)

    reader_thread = threading.Thread(target=serial_reader, args=(ser,), daemon=True)
    reader_thread.start()

    print(
        f"Listening on {SERIAL_PORT}. "
        f"Press SPACEBAR to start logging ({SAMPLES_PER_SESSION} samples). "
        f"Press SPACEBAR again to stop early. Press ESC to quit.\n"
    )

    last_toggle_time = 0

    try:
        while True:
            if session_ready.is_set():
                session_ready.clear()
                prompt_and_save_session()
                last_toggle_time = time.time()
                flush_keyboard_buffer()

            if msvcrt.kbhit():
                key = msvcrt.getch()
                now = time.time()

                if key == b" ":
                    if now - last_toggle_time < DEBOUNCE_SECONDS:
                        continue
                    last_toggle_time = now
                    flush_keyboard_buffer()

                    if not logging_active.is_set():
                        session_ready.clear()
                        with buffer_lock:
                            buffer.clear()
                        logging_active.set()
                        print(f"\nLogging started (target: {SAMPLES_PER_SESSION} samples)...\n")
                    else:
                        logging_active.clear()
                        prompt_and_save_session()
                        last_toggle_time = time.time()
                        flush_keyboard_buffer()

                elif key == b"\x1b":
                    if logging_active.is_set():
                        logging_active.clear()
                        print("\nLogging stopped (session discarded, exiting).")
                    print("Exiting.")
                    break

            time.sleep(0.01)

    except KeyboardInterrupt:
        pass
    finally:
        stop_flag = True
        ser.close()


if __name__ == "__main__":
    main()
