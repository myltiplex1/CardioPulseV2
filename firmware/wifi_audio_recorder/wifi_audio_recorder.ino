#include <driver/i2s.h>
#include <WiFi.h>

#define I2S_WS 15
#define I2S_SD 16
#define I2S_SCK 17
#define I2S_PORT I2S_NUM_0

// --- EDIT THESE if you want ---
const char* AP_SSID = "HeartMonitor_ESP32";
const char* AP_PASSWORD = "heartbeat123";  // min 8 chars, WPA2 requirement
const uint16_t TCP_PORT = 3333;
// The ESP32's own AP IP is always 192.168.4.1 by default -- use this in
// the Python receiver script. Connect your PC/Pi's WiFi to AP_SSID first,
// the same way you'd join any WiFi network, before running the script.

#define I2S_READ_BLOCK_SAMPLES 256
int32_t i2s_buf[I2S_READ_BLOCK_SAMPLES];
int16_t out_buf[I2S_READ_BLOCK_SAMPLES];

WiFiServer server(TCP_PORT);
WiFiClient client;
bool isRecording = false;

void setup() {
  Serial.begin(115200);  // debug/status only -- audio data goes over WiFi, not this

  WiFi.mode(WIFI_AP);
  WiFi.softAP(AP_SSID, AP_PASSWORD);
  WiFi.setSleep(false);  // same fix as before -- no reason to skip it here

  Serial.print("Access point started. SSID: ");
  Serial.println(AP_SSID);
  Serial.print("ESP32 AP IP address: ");
  Serial.println(WiFi.softAPIP());  // normally 192.168.4.1
  Serial.println("Connect your PC/Pi's WiFi to this network, then run the receiver script.");

  server.begin();

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
  if (!client || !client.connected()) {
    client = server.available();
    if (client) {
      client.setNoDelay(true);
      Serial.println("Receiver connected.");
    }
    return;
  }

  if (client.available() > 0) {
    char command = client.read();
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

      client.write((uint8_t*)out_buf, samples_read * sizeof(int16_t));
    }
  }
}
