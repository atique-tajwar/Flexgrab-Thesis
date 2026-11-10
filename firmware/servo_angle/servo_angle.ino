/*
  servo_angle.ino  -  move the servo to any angle from the Serial Monitor
  Board: classic ESP32 DevKit v1 (Arduino core 2.x or 3.x, no extra library)

  Serial Monitor: 115200 baud, line ending "Newline" (or "Both NL & CR")

    90      + Enter  -> go to 90 deg      (decimals OK: 45.5)
    space   + Enter  -> go back to 0 deg
    (in PuTTY / a terminal that sends keys instantly, space alone works)

  Servo wiring: signal -> SERVO_PIN, servo + -> 6 V supply (NOT the ESP32),
  servo - -> supply GND AND ESP32 GND (common ground).
*/

#include <Arduino.h>

// ---------- settings ----------
const int   SERVO_PIN = 9;      // change to the pin your servo signal is on
const float PULSE_MIN = 500.0;   // us at 0 deg   (DS3235 / FT5330M: 500-2500)
const float PULSE_MAX = 2500.0;  // us at FULL_ANGLE
const float FULL_ANGLE = 180.0;  // servo's full travel in deg

// Safety limits for the gripper: keep the fingers off their hard stops.
// Start wide, then tighten once you know where the fingers hit.
const float ANGLE_MIN = 0.0;
const float ANGLE_MAX = 180.0;
// ------------------------------

const int SERVO_FREQ = 50;       // Hz -> 20 000 us period
const int SERVO_RES  = 16;       // bits -> 0.305 us per count
const int LEDC_CH    = 0;        // only used on core 2.x

String buf = "";
float currentAngle = 0.0;

uint32_t usToDuty(float us) {
  const float period_us = 1000000.0 / SERVO_FREQ;
  return (uint32_t)(us / period_us * ((1UL << SERVO_RES) - 1) + 0.5);
}

void writeAngle(float deg) {
  if (deg < ANGLE_MIN) { Serial.printf("  limited to ANGLE_MIN %.1f\n", ANGLE_MIN); deg = ANGLE_MIN; }
  if (deg > ANGLE_MAX) { Serial.printf("  limited to ANGLE_MAX %.1f\n", ANGLE_MAX); deg = ANGLE_MAX; }
  float us = PULSE_MIN + (PULSE_MAX - PULSE_MIN) * deg / FULL_ANGLE;
  uint32_t duty = usToDuty(us);
#if ESP_ARDUINO_VERSION_MAJOR >= 3
  ledcWrite(SERVO_PIN, duty);
#else
  ledcWrite(LEDC_CH, duty);
#endif
  currentAngle = deg;
  Serial.printf("-> %.2f deg  (%.1f us, duty %lu)\n", deg, us, (unsigned long)duty);
}

void setup() {
  Serial.begin(115200);
  delay(300);
#if ESP_ARDUINO_VERSION_MAJOR >= 3
  ledcAttach(SERVO_PIN, SERVO_FREQ, SERVO_RES);
#else
  ledcSetup(LEDC_CH, SERVO_FREQ, SERVO_RES);
  ledcAttachPin(SERVO_PIN, LEDC_CH);
#endif
  Serial.println();
  Serial.println("Servo angle control");
  Serial.println("  type an angle + Enter      e.g. 90");
  Serial.println("  space + Enter              back to 0");
  writeAngle(0.0);
}

void loop() {
  while (Serial.available()) {
    char c = Serial.read();

    if (c == ' ' && buf.length() == 0) {   // space on its own -> home
      Serial.println("[space] home");
      writeAngle(0.0);
      continue;
    }

    if (c == '\n' || c == '\r') {
      buf.trim();
      if (buf.length() > 0) {
        char *end;
        float a = strtof(buf.c_str(), &end);
        if (end != buf.c_str()) writeAngle(a);
        else Serial.printf("  '%s' is not a number\n", buf.c_str());
      }
      buf = "";
      continue;
    }

    if (buf.length() < 16) buf += c;
  }
}
