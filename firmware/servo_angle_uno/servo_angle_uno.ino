/*
  servo_angle_uno.ino  -  move the servo to any angle from the Serial Monitor
  Board: Arduino Uno / Nano (uses the built-in Servo library)

  Serial Monitor: 115200 baud, line ending "Newline"
    90     + Enter  -> go to 90 deg   (decimals OK: 45.5)
    space  + Enter  -> back to 0 deg

  Wiring: signal -> pin 9, servo + -> external 5-6 V supply (NOT the Uno 5V pin),
  servo - -> supply GND AND Uno GND (common ground).
*/

#include <Servo.h>

const int   SERVO_PIN  = 9;
const float PULSE_MIN  = 500.0;   // us at 0 deg
const float PULSE_MAX  = 2500.0;  // us at FULL_ANGLE
const float FULL_ANGLE = 180.0;

const float ANGLE_MIN = 0.0;      // tighten once you know the safe range
const float ANGLE_MAX = 180.0;

Servo servo;
String buf = "";

void writeAngle(float deg) {
  if (deg < ANGLE_MIN) deg = ANGLE_MIN;
  if (deg > ANGLE_MAX) deg = ANGLE_MAX;
  int us = (int)(PULSE_MIN + (PULSE_MAX - PULSE_MIN) * deg / FULL_ANGLE + 0.5);
  servo.writeMicroseconds(us);
  Serial.print("-> ");
  Serial.print(deg, 2);
  Serial.print(" deg  (");
  Serial.print(us);
  Serial.println(" us)");
}

void setup() {
  Serial.begin(115200);
  servo.attach(SERVO_PIN, (int)PULSE_MIN, (int)PULSE_MAX);
  Serial.println(F("Servo angle control: angle + Enter, space + Enter = home"));
  writeAngle(0.0);
}

void loop() {
  while (Serial.available()) {
    char c = Serial.read();

    if (c == ' ' && buf.length() == 0) {   // space on its own -> home
      Serial.println(F("[space] home"));
      writeAngle(0.0);
      continue;
    }

    if (c == '\n' || c == '\r') {
      buf.trim();
      if (buf.length() > 0) {
        char first = buf.charAt(0);
        if ((first >= '0' && first <= '9') || first == '.' || first == '-') {
          writeAngle(buf.toFloat());
        } else {
          Serial.print(F("  not a number: "));
          Serial.println(buf);
        }
      }
      buf = "";
      continue;
    }

    if (buf.length() < 16) buf += c;
  }
}
