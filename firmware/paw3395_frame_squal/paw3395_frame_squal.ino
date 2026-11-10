/*
 * PAW3395 Characterization Logger + Raw Frame Capture (with SQUAL)
 * ===============================================================
 * Same as paw3395_logger.ino, plus an 'F' command that returns one
 * 36x36 raw pixel frame (used by sensor_image_frame.py).
 * Board: classic ESP32 DevKit v1 (VSPI, GPIO 18/19/23/5, NRESET on 4).
 * ESP32-S3 firmware for characterizing the PAW3395 optical sensor prior to
 * integration into the gripper.
 *
 * Differences from the earlier sketch, and why:
 *   - Uses DPI_Config() from PAW3395.h instead of writing 0x4D/0x4E.
 *     Those are NOT the resolution registers. The header defines them as
 *     SET_RESOLUTION 0x47, RESOLUTION_X_LOW/HIGH 0x48/0x49,
 *     RESOLUTION_Y_LOW/HIGH 0x4A/0x4B -- and resolution is 16-bit, so a
 *     single 8-bit write could not have set it in any case.
 *   - Uses Motion_Burst() instead of five separate register reads. The
 *     separate reads are non-atomic: the sensor can update between them,
 *     pairing a dx from one sample with a dy from the next.
 *   - Emits a microsecond timestamp with every sample. Slip velocity is
 *     displacement / dt; without a real dt the velocity is unknown.
 *   - Logs EVERY poll, not only polls where the motion bit is set, so a
 *     stationary noise floor can actually be measured.
 *   - Fixed-interval polling against a micros() deadline, so dt is constant
 *     and known rather than set by delay() plus serial print time.
 *
 * Serial commands (newline-terminated):
 *   c<n>    set CPI, e.g.  c16000
 *   p<n>    set poll interval in microseconds, e.g.  p2000
 *   s       start streaming
 *   x       stop streaming
 *   i       print current configuration
 *   F       capture one raw 36x36 frame: sends FF FF AA 55, SQUAL (1 byte),
 *           shutter (2 bytes, little-endian), then 1296 pixel bytes
 *           (stops streaming; keep the sensor still while capturing)
 *
 * Output format (CSV, one line per poll):
 *   t_us,dx,dy,motion
 *
 * NOTE: 921600 baud. At a 1 ms poll interval this stream is ~25 kB/s, which
 * exceeds what 115200 baud can carry; lines would be dropped silently.
 */

#include <Arduino.h>
#include "PAW3395.h"
#include "usr_spi.h"

#define PIN_NRESET   4
#define SERIAL_BAUD  921600

// ---- configuration (changeable at runtime over serial) ----
uint16_t cpi          = 12800;   // see note below on CPI selection
uint32_t pollInterval = 2000;    // microseconds
bool     streaming    = false;

/*
 * CPI selection note:
 * At 800 CPI and a 2 ms poll, a 10 mm/s slip produces 0.6 counts -- entirely
 * quantization-dominated. Roughly 5 counts per poll at the slowest slip you
 * intend to detect is a sensible target, which puts you at 10000-16000 CPI.
 * The distance-calibration run verifies both the CPI setting and the
 * semantics of DPI_Config().
 */

uint32_t nextPoll = 0;

void printConfig() {
  Serial.print(F("# cpi=")); Serial.print(cpi);
  Serial.print(F(" poll_us=")); Serial.print(pollInterval);
  Serial.print(F(" count_um="));
  Serial.print(25400.0f / cpi, 3);
  Serial.print(F(" v_quant_mm_s="));
  Serial.println((25.4f / cpi) / (pollInterval * 1e-6f), 3);
}

void applyCPI(uint16_t v) {
  cpi = v;
  DPI_Config(cpi);          // declared in PAW3395.h
  delayMicroseconds(100);
}

void handleSerial() {
  if (!Serial.available()) return;
  String cmd = Serial.readStringUntil('\n');
  cmd.trim();
  if (cmd.length() == 0) return;

  char c = cmd.charAt(0);
  String arg = cmd.substring(1);

  switch (c) {
    case 'c': {
      long v = arg.toInt();
      if (v >= 50 && v <= 26000) { applyCPI((uint16_t)v); printConfig(); }
      else Serial.println(F("# ERR cpi out of range (50-26000)"));
      break;
    }
    case 'p': {
      long v = arg.toInt();
      if (v >= 250 && v <= 100000) { pollInterval = (uint32_t)v; printConfig(); }
      else Serial.println(F("# ERR poll interval out of range (250-100000 us)"));
      break;
    }
    case 's':
      streaming = true;
      nextPoll  = micros();
      Serial.println(F("# STREAM START"));
      Serial.println(F("t_us,dx,dy,motion,squal,shutter"));   // was 4 columns
      break;
    case 'x':
      streaming = false;
      Serial.println(F("# STREAM STOP"));
      break;
    case 'i':
      printConfig();
      break;
    case 'F': {
      // One motion burst first, so the frame comes with the SQUAL and
      // shutter the sensor reports at that moment. Then the raw frame.
      // Pixel_Burst_Read() stores pixel k at index 2*k, so it needs a
      // buffer twice the frame size; pack to 1296 bytes before sending.
      // Packet: FF FF AA 55 | squal (1) | shutter lo, hi (2) | 1296 pixels
      streaming = false;
      int16_t fdx = 0, fdy = 0;
      uint8_t fsq = 0;
      uint16_t fsh = 0;
      Motion_Burst_Ext(&fdx, &fdy, &fsq, &fsh);
      static uint8_t raw[36 * 36 * 2];
      static uint8_t frame[36 * 36];
      Pixel_Burst_Read(raw);
      for (int i = 0; i < 36 * 36; i++) frame[i] = raw[2 * i];
      const uint8_t hdr[7] = {0xFF, 0xFF, 0xAA, 0x55, fsq,
                              (uint8_t)(fsh & 0xFF), (uint8_t)(fsh >> 8)};
      Serial.write(hdr, sizeof(hdr));
      Serial.write(frame, sizeof(frame));
      break;
    }
    default:
      Serial.println(F("# ERR unknown command"));
  }
}

void setup() {
  Serial.begin(SERIAL_BAUD);
  delay(100);

  // Chip select idle high before any bus activity
  pinMode(NCS, OUTPUT);
  digitalWrite(NCS, HIGH);

  // Hardware reset. The datasheet requires NRESET be driven continuously
  // by a microcontroller pin rather than left floating.
  pinMode(PIN_NRESET, OUTPUT);
  digitalWrite(PIN_NRESET, LOW);
  delay(50);
  digitalWrite(PIN_NRESET, HIGH);
  delay(50);

  paw3395Init();     // power-up sequence + SROM upload
  delay(10);
  applyCPI(cpi);

  Serial.println(F("# PAW3395 characterization logger"));
  printConfig();
  Serial.println(F("# commands: c<cpi> p<us> s(tart) x(stop) i(nfo)"));
}

void loop() {
  handleSerial();
  if (!streaming) return;

  uint32_t now = micros();
  if ((int32_t)(now - nextPoll) < 0) return;   // wraparound-safe compare
  nextPoll += pollInterval;

  // Guard against falling behind: if we have missed deadlines (e.g. the
  // serial buffer stalled), resynchronise rather than burst-catching-up,
  // which would corrupt the uniform sampling interval.
  if ((int32_t)(micros() - nextPoll) > (int32_t)pollInterval) {
    nextPoll = micros() + pollInterval;
  }

  int16_t dx = 0, dy = 0;
  uint8_t squal = 0;
  uint16_t shutter = 0;
  Motion_Burst_Ext(&dx, &dy, &squal, &shutter);

  uint8_t motion = (dx != 0 || dy != 0) ? 1 : 0;   // this line was missing

  Serial.print(now);     Serial.print(',');
  Serial.print(dx);      Serial.print(',');
  Serial.print(dy);      Serial.print(',');
  Serial.print(motion);  Serial.print(',');
  Serial.print(squal);   Serial.print(',');
  Serial.println(shutter);
}

