#include <Wire.h>

// Reads raw accelerometer + gyroscope values from the MPU6050 and prints
// them, so we can confirm sensible readings before building any real
// motion-detection logic on top. Hold the board still and check the
// values make sense (see expected ranges printed below) before moving on.

#define SDA_PIN 8
#define SCL_PIN 9
#define MPU6050_ADDR 0x68

// Key MPU6050 registers
#define PWR_MGMT_1   0x6B
#define ACCEL_XOUT_H 0x3B

void writeRegister(uint8_t reg, uint8_t value) {
  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(reg);
  Wire.write(value);
  Wire.endTransmission();
}

void setup() {
  Serial.begin(115200);
  while (!Serial) { delay(10); }

  Wire.begin(SDA_PIN, SCL_PIN);

  // Wake the MPU6050 up -- it starts in sleep mode by default
  writeRegister(PWR_MGMT_1, 0x00);
  delay(100);

  Serial.println("MPU6050 initialized.");
  Serial.println("Expected at rest, flat: accel Z ~1.0g, X/Y ~0.0g, gyro ~0 deg/s on all axes.");
  Serial.println();
}

void loop() {
  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(ACCEL_XOUT_H);
  Wire.endTransmission(false);
  Wire.requestFrom(MPU6050_ADDR, 14, true);  // 14 bytes: accel(6) + temp(2) + gyro(6)

  int16_t ax_raw = Wire.read() << 8 | Wire.read();
  int16_t ay_raw = Wire.read() << 8 | Wire.read();
  int16_t az_raw = Wire.read() << 8 | Wire.read();
  int16_t temp_raw = Wire.read() << 8 | Wire.read();
  int16_t gx_raw = Wire.read() << 8 | Wire.read();
  int16_t gy_raw = Wire.read() << 8 | Wire.read();
  int16_t gz_raw = Wire.read() << 8 | Wire.read();

  // Default sensitivity: accel +/-2g range -> divide by 16384 for g's
  // gyro +/-250 deg/s range -> divide by 131 for deg/s
  float ax = ax_raw / 16384.0;
  float ay = ay_raw / 16384.0;
  float az = az_raw / 16384.0;
  float gx = gx_raw / 131.0;
  float gy = gy_raw / 131.0;
  float gz = gz_raw / 131.0;
  float temp_c = temp_raw / 340.0 + 36.53;

  float magnitude = sqrt(ax * ax + ay * ay + az * az);

  Serial.print("Accel (g): X=");
  Serial.print(ax, 3);
  Serial.print(" Y=");
  Serial.print(ay, 3);
  Serial.print(" Z=");
  Serial.print(az, 3);
  Serial.print(" | Magnitude=");
  Serial.print(magnitude, 3);

  Serial.print("  Gyro (deg/s): X=");
  Serial.print(gx, 2);
  Serial.print(" Y=");
  Serial.print(gy, 2);
  Serial.print(" Z=");
  Serial.print(gz, 2);

  Serial.print("  Temp: ");
  Serial.print(temp_c, 1);
  Serial.println(" C");

  delay(200);
}
