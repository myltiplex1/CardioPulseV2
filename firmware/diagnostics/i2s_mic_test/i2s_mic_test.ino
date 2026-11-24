#include <driver/i2s.h>
#include <WiFi.h>
#include <esp_bt.h>

#define I2S_WS 15
#define I2S_SD 16
#define I2S_SCK 17
#define I2S_PORT I2S_NUM_0
#define SERIAL_BAUD 115200

#define BLOCK_SAMPLES 256
int32_t i2s_buf[BLOCK_SAMPLES];

// A sane INMP441 sample after >>14 should stay well within 16-bit range.
// Anything beyond this is almost certainly corruption, not real audio.
#define SANE_THRESHOLD 40000

void setup() {
  Serial.begin(SERIAL_BAUD);
  delay(500);

  // Disable WiFi and Bluetooth radios -- both are known noise sources
  // that can couple into nearby digital wiring on a breadboard.
  WiFi.mode(WIFI_OFF);
  btStop();

  i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = 16000,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 8,
    .dma_buf_len = 64,
    .use_apll = false,
    .tx_desc_auto_clear = false,
    .fixed_mclk = 0
  };

  i2s_pin_config_t pin_config = {
    .bck_io_num = I2S_SCK,
    .ws_io_num = I2S_WS,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num = I2S_SD
  };

  i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
  i2s_set_pin(I2S_PORT, &pin_config);

  Serial.println("Diagnostic started (WiFi/BT disabled).");
  Serial.println("Hold the mic still in a quiet room for this test.");
}

void loop() {
  static int32_t raw_min = INT32_MAX;
  static int32_t raw_max = INT32_MIN;
  static int64_t raw_sum = 0;
  static uint32_t sample_count = 0;
  static uint32_t spike_count = 0;
  static unsigned long last_print = 0;

  size_t bytes_read = 0;
  esp_err_t result = i2s_read(I2S_PORT, i2s_buf, sizeof(i2s_buf),
                               &bytes_read, portMAX_DELAY);

  if (result == ESP_OK && bytes_read > 0) {
    int n = bytes_read / sizeof(int32_t);
    for (int i = 0; i < n; i++) {
      int32_t v = i2s_buf[i];
      if (v < raw_min) raw_min = v;
      if (v > raw_max) raw_max = v;
      raw_sum += v;
      sample_count++;

      int32_t shifted = v >> 14;
      if (abs(shifted) > SANE_THRESHOLD) spike_count++;
    }
  }

  if (millis() - last_print > 1000) {
    last_print = millis();
    if (sample_count > 0) {
      float mean = (float)raw_sum / sample_count;
      float spike_pct = 100.0f * spike_count / sample_count;
      Serial.println("---- 1s window ----");
      Serial.printf("Raw 32-bit min: %ld\n", (long)raw_min);
      Serial.printf("Raw 32-bit max: %ld\n", (long)raw_max);
      Serial.printf("Raw 32-bit mean: %.1f\n", mean);
      Serial.printf("After >>14 shift -- min: %ld  max: %ld\n",
                    (long)(raw_min >> 14), (long)(raw_max >> 14));
      Serial.printf("Corrupted/out-of-range samples: %lu / %lu (%.2f%%)\n",
                    (unsigned long)spike_count, (unsigned long)sample_count, spike_pct);
    }
    raw_min = INT32_MAX;
    raw_max = INT32_MIN;
    raw_sum = 0;
    sample_count = 0;
    spike_count = 0;
  }
}
