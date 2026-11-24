#include <driver/i2s.h>
#include <WiFi.h>
#include <Wire.h>

#define I2S_WS 15
#define I2S_SD 16
#define I2S_SCK 17
#define I2S_PORT I2S_NUM_0

// --- MPU6050 (I2C) ---
#define SDA_PIN 8
#define SCL_PIN 9
#define MPU6050_ADDR 0x68
#define PWR_MGMT_1   0x6B
#define ACCEL_XOUT_H 0x3B
const unsigned long MOTION_READ_INTERVAL_MS = 20;  // 50Hz -- plenty for
                                                    // magnitude/variance,
                                                    // deliberately modest
                                                    // so it never competes
                                                    // meaningfully with audio
unsigned long last_motion_read = 0;

// --- EDIT THESE if you want ---
const char* AP_SSID = "HeartMonitor_ESP32";
const char* AP_PASSWORD = "heartbeat123";
const uint16_t TCP_PORT_AUDIO = 3333;
const uint16_t TCP_PORT_MOTION = 3334;  // separate port -- audio path
                                         // stays completely untouched
                                         // regardless of anything here

#define I2S_READ_BLOCK_SAMPLES 256
int32_t i2s_buf[I2S_READ_BLOCK_SAMPLES];
int16_t out_buf[I2S_READ_BLOCK_SAMPLES];

WiFiServer audioServer(TCP_PORT_AUDIO);
WiFiClient audioClient;
bool isRecording = false;

WiFiServer motionServer(TCP_PORT_MOTION);
WiFiClient motionClient;

void writeRegister(uint8_t reg, uint8_t value) {
  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(reg);
  Wire.write(value);
  Wire.endTransmission();
}

void setup() {
  Serial.begin(115200);

  WiFi.mode(WIFI_AP);
  WiFi.softAP(AP_SSID, AP_PASSWORD);
  WiFi.setSleep(false);

  Serial.print("Access point started. SSID: ");
  Serial.println(AP_SSID);
  Serial.print("ESP32 AP IP address: ");
  Serial.println(WiFi.softAPIP());
  Serial.println("Connect your PC/Pi's WiFi to this network, then run the receiver script.");

  audioServer.begin();
  motionServer.begin();

  // --- I2S (audio) setup -- unchanged from before ---
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

  // --- I2C (motion) setup -- separate bus, separate pins, independent ---
  Wire.begin(SDA_PIN, SCL_PIN);
  writeRegister(PWR_MGMT_1, 0x00);  // wake the MPU6050 up
  delay(100);
  Serial.println("MPU6050 initialized on separate I2C bus.");
}

void handleAudio() {
  // Exactly the same logic as before -- untouched.
  if (!audioClient || !audioClient.connected()) {
    audioClient = audioServer.available();
    if (audioClient) {
      audioClient.setNoDelay(true);
      Serial.println("Audio receiver connected.");
    }
    return;
  }

  if (audioClient.available() > 0) {
    char command = audioClient.read();
    if (command == 'r') { isRecording = true; Serial.println("Recording started."); }
    if (command == 's') { isRecording = false; Serial.println("Recording stopped."); }
  }

  if (isRecording) {
    size_t bytes_read = 0;
    esp_err_t result = i2s_read(I2S_PORT, i2s_buf, sizeof(i2s_buf),
                                 &bytes_read, portMAX_DELAY);

    if (result == ESP_OK && bytes_read > 0) {
      int samples_read = bytes_read / sizeof(int32_t);

      for (int i = 0; i < samples_read; i++) {
        int32_t audio_data = (i2s_buf[i] >> 14) * 1;
        if (audio_data > 32767) audio_data = 32767;
        if (audio_data < -32768) audio_data = -32768;
        out_buf[i] = (int16_t)audio_data;
      }

      audioClient.write((uint8_t*)out_buf, samples_read * sizeof(int16_t));
    }
  }
}

void handleMotion() {
  if (!motionClient || !motionClient.connected()) {
    motionClient = motionServer.available();
    if (motionClient) {
      motionClient.setNoDelay(true);
      Serial.println("Motion receiver connected.");
    }
    return;
  }

  // Non-blocking timing check -- only reads at MOTION_READ_INTERVAL_MS,
  // never delays or blocks the main loop (and therefore never competes
  // with the audio path above).
  unsigned long now = millis();
  if (now - last_motion_read < MOTION_READ_INTERVAL_MS) {
    return;
  }
  last_motion_read = now;

  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(ACCEL_XOUT_H);
  Wire.endTransmission(false);
  Wire.requestFrom(MPU6050_ADDR, 6, true);  // just accel X/Y/Z -- 6 bytes,
                                            // gyro not needed for the
                                            // magnitude-based motion score

  if (Wire.available() < 6) return;

  int16_t ax_raw = Wire.read() << 8 | Wire.read();
  int16_t ay_raw = Wire.read() << 8 | Wire.read();
  int16_t az_raw = Wire.read() << 8 | Wire.read();

  // Send raw int16 values -- same lightweight pattern as the audio path.
  // Scaling to g's happens on the receiver side (Python), same
  // "keep firmware simple, do intelligence on the receiving end" pattern
  // used throughout the rest of this project.
  int16_t motion_packet[3] = {ax_raw, ay_raw, az_raw};
  motionClient.write((uint8_t*)motion_packet, sizeof(motion_packet));
}

void loop() {
  handleAudio();
  handleMotion();
}
