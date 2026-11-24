#include <Wire.h>

// Standalone I2C scanner -- run this BEFORE writing any real MPU6050
// code. It just checks whether anything responds on the I2C bus, and at
// what address. The MPU6050 should show up at 0x68 (default, AD0 tied
// to GND or left floating) or 0x69 (if AD0 is tied to 3.3V).

#define SDA_PIN 8
#define SCL_PIN 9

void setup() {
  Serial.begin(115200);
  while (!Serial) { delay(10); }

  Wire.begin(SDA_PIN, SCL_PIN);

  Serial.println();
  Serial.println("I2C Scanner starting...");
}

void loop() {
  byte error, address;
  int devices_found = 0;

  Serial.println("Scanning...");

  for (address = 1; address < 127; address++) {
    Wire.beginTransmission(address);
    error = Wire.endTransmission();

    if (error == 0) {
      Serial.print("Device found at address 0x");
      if (address < 16) Serial.print("0");
      Serial.println(address, HEX);
      devices_found++;

      if (address == 0x68 || address == 0x69) {
        Serial.println("  -> This matches the expected MPU6050 address.");
      }
    } else if (error == 4) {
      Serial.print("Unknown error at address 0x");
      if (address < 16) Serial.print("0");
      Serial.println(address, HEX);
    }
  }

  if (devices_found == 0) {
    Serial.println("No I2C devices found. Check wiring:");
    Serial.println("  VCC -> 3.3V");
    Serial.println("  GND -> GND");
    Serial.println("  SCL -> GPIO22");
    Serial.println("  SDA -> GPIO21");
  } else {
    Serial.print(devices_found);
    Serial.println(" device(s) found.");
  }

  Serial.println();
  delay(3000);
}
