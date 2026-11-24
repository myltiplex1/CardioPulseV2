#include <driver/i2s.h>

#define I2S_WS 15
#define I2S_SD 16
#define I2S_SCK 17
#define I2S_PORT I2S_NUM_0

// Must match BAUD_RATE in the Python receiver script exactly.
#define SERIAL_BAUD 921600

// Block size for I2S reads. Reading in blocks instead of one sample
// at a time is more robust against timing hiccups.
#define I2S_READ_BLOCK_SAMPLES 256

bool isRecording = false;
int32_t i2s_buf[I2S_READ_BLOCK_SAMPLES];
int16_t out_buf[I2S_READ_BLOCK_SAMPLES];

void setup() {
  Serial.begin(SERIAL_BAUD);

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
}

void loop() {
  if (Serial.available() > 0) {
    char command = Serial.read();
    if (command == 'r') isRecording = true;
    if (command == 's') isRecording = false;
  }

  if (isRecording) {
    size_t bytes_read = 0;
    esp_err_t result = i2s_read(I2S_PORT, i2s_buf,
                                 sizeof(i2s_buf), &bytes_read,
                                 portMAX_DELAY);

    if (result == ESP_OK && bytes_read > 0) {
      int samples_read = bytes_read / sizeof(int32_t);

      for (int i = 0; i < samples_read; i++) {
        // Shift 32-bit I2S frame down to useful range.
        // Gain left at 1x on purpose — do gain adjustments in
        // software later where you can see the waveform and avoid
        // clipping. If you truly need more gain here, increase this
        // slowly (e.g. 2, then 4) and watch for clipped/flat peaks.
        int32_t audio_data = (i2s_buf[i] >> 14) * 1;

        if (audio_data > 32767) audio_data = 32767;
        if (audio_data < -32768) audio_data = -32768;

        out_buf[i] = (int16_t)audio_data;
      }

      Serial.write((uint8_t*)out_buf, samples_read * sizeof(int16_t));
    }
  }
}
