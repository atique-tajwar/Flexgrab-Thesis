/*
 * FSR stream for calibration and repeatability tests
 * ==================================================
 * Board : classic ESP32 DevKit v1 (FSR divider on GPIO 34, ADC1)
 * Circuit: 3.3 V -> FSR -> node -> R_FIXED -> GND ; node -> GPIO 34
 *          10 nF (103) ceramic from GPIO 34 to GND, close to the board
 *
 * Every 50 ms (20 Hz) prints one CSV line:
 *     t_ms,adc_raw,mv,r_ohm
 *   t_ms   ESP32 time stamp (real sample timing)
 *   adc_raw averaged raw ADC count (0..4095)
 *   mv     averaged pin voltage in mV, from analogReadMilliVolts(), which
 *          applies the chip's factory ADC calibration (more accurate than
 *          raw*3.3/4095 on the classic ESP32, whose ADC is non-linear)
 *   r_ohm  FSR resistance; -1 when there is no force (open circuit)
 *
 * Each value averages 20 samples spaced 1 ms apart = a 20 ms window,
 * which cancels 50 Hz mains hum.
 *
 * Lines starting with '#' are comments for the logger.
 */

const int   FSR_PIN    = 13;
const float VCC_MV     = 3340.0;   // measure your 3V3 rail and put the real value here
const float R_FIXED    = 5050.0;   // MEASURE your 5 k resistor and put the real value here
const int   N_AVG      = 20;       // 20 x 1 ms = 20 ms window (rejects 50 Hz)
const unsigned long PERIOD_MS = 50; // 20 lines per second

unsigned long next_t = 0;

void setup() {
  Serial.begin(115200);
  analogReadResolution(12);
  analogSetPinAttenuation(FSR_PIN, ADC_11db);
  delay(200);
  Serial.println("# fsr_stream: t_ms,adc_raw,mv,r_ohm");
  Serial.printf("# R_FIXED=%.1f VCC_MV=%.1f N_AVG=%d PERIOD_MS=%lu\n",
                R_FIXED, VCC_MV, N_AVG, PERIOD_MS);
  next_t = millis();
}

void loop() {
  if ((long)(millis() - next_t) < 0) return;
  unsigned long t0 = millis();
  next_t += PERIOD_MS;

  long sum_raw = 0, sum_mv = 0;
  for (int i = 0; i < N_AVG; i++) {
    sum_raw += analogRead(FSR_PIN);
    sum_mv  += analogReadMilliVolts(FSR_PIN);
    delayMicroseconds(1000 - 60);   // ~1 ms per sample incl. conversion time
  }
  float raw = (float)sum_raw / N_AVG;
  float mv  = (float)sum_mv  / N_AVG;

  float r = -1;
  if (mv > 20.0 && mv < VCC_MV - 5.0) {
    r = (VCC_MV - mv) * R_FIXED / mv;
  }
  Serial.printf("%lu,%.1f,%.1f,%.1f\n", t0, raw, mv, r);
}
