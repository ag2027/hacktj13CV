/*
  Hybrid Rover Controller (Arduino Side)
  --------------------------------------
  Board: Arduino Uno
  Sensors: HC-SR04 ultrasonic sensor
  Driver: L298N motor driver

  Protocol:
    - Outgoing sensor line: DIST:<value_cm>
    - Incoming commands: FORWARD, LEFT, RIGHT, STOP
*/

// ----------------------------
// Pin assignments (assumptions)
// ----------------------------
const int TRIG_PIN = 9;
const int ECHO_PIN = 10;

const int ENA_PIN = 5;   // PWM left motor enable
const int IN1_PIN = 2;   // Left motor direction 1
const int IN2_PIN = 3;   // Left motor direction 2

const int ENB_PIN = 6;   // PWM right motor enable
const int IN3_PIN = 7;   // Right motor direction 1
const int IN4_PIN = 8;   // Right motor direction 2

// ----------------------------
// Control parameters
// ----------------------------
const int DRIVE_SPEED = 170;                // 0-255
const unsigned long SENSOR_PERIOD_MS = 120; // Sensor publish rate
const unsigned long CMD_TIMEOUT_MS = 1000;  // Safety timeout

// ----------------------------
// Runtime state
// ----------------------------
unsigned long lastSensorTime = 0;
unsigned long lastCommandTime = 0;

char cmdBuffer[20];
byte cmdIndex = 0;

// ----------------------------
// Motor helpers
// ----------------------------
void moveForward() {
  analogWrite(ENA_PIN, DRIVE_SPEED);
  analogWrite(ENB_PIN, DRIVE_SPEED);

  digitalWrite(IN1_PIN, HIGH);
  digitalWrite(IN2_PIN, LOW);
  digitalWrite(IN3_PIN, HIGH);
  digitalWrite(IN4_PIN, LOW);
}

void turnLeft() {
  analogWrite(ENA_PIN, DRIVE_SPEED);
  analogWrite(ENB_PIN, DRIVE_SPEED);

  // Pivot left: left wheel backward, right wheel forward
  digitalWrite(IN1_PIN, LOW);
  digitalWrite(IN2_PIN, HIGH);
  digitalWrite(IN3_PIN, HIGH);
  digitalWrite(IN4_PIN, LOW);
}

void turnRight() {
  analogWrite(ENA_PIN, DRIVE_SPEED);
  analogWrite(ENB_PIN, DRIVE_SPEED);

  // Pivot right: left wheel forward, right wheel backward
  digitalWrite(IN1_PIN, HIGH);
  digitalWrite(IN2_PIN, LOW);
  digitalWrite(IN3_PIN, LOW);
  digitalWrite(IN4_PIN, HIGH);
}

void stopMotors() {
  // Disable both motors for a hard stop
  analogWrite(ENA_PIN, 0);
  analogWrite(ENB_PIN, 0);

  digitalWrite(IN1_PIN, LOW);
  digitalWrite(IN2_PIN, LOW);
  digitalWrite(IN3_PIN, LOW);
  digitalWrite(IN4_PIN, LOW);
}

// Returns distance in cm. If no echo, returns 400 cm as a "far" reading.
long readDistanceCM() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  unsigned long duration = pulseIn(ECHO_PIN, HIGH, 25000UL); // ~4m max range
  if (duration == 0) {
    return 400;
  }
  return (long)(duration * 0.0343 / 2.0);
}

void processCommand(const char* command) {
  if (strcmp(command, "FORWARD") == 0) {
    moveForward();
    lastCommandTime = millis();
  } else if (strcmp(command, "LEFT") == 0) {
    turnLeft();
    lastCommandTime = millis();
  } else if (strcmp(command, "RIGHT") == 0) {
    turnRight();
    lastCommandTime = millis();
  } else if (strcmp(command, "STOP") == 0) {
    stopMotors();
    lastCommandTime = millis();
  }
}

// Non-blocking serial line reader.
void pollSerial() {
  while (Serial.available() > 0) {
    char c = (char)Serial.read();

    if (c == '\n' || c == '\r') {
      if (cmdIndex > 0) {
        cmdBuffer[cmdIndex] = '\0';
        processCommand(cmdBuffer);
        cmdIndex = 0;
      }
    } else if (cmdIndex < sizeof(cmdBuffer) - 1) {
      cmdBuffer[cmdIndex++] = c;
    } else {
      // Buffer overflow guard: reset and wait for a fresh line.
      cmdIndex = 0;
    }
  }
}

void setup() {
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);

  pinMode(ENA_PIN, OUTPUT);
  pinMode(IN1_PIN, OUTPUT);
  pinMode(IN2_PIN, OUTPUT);
  pinMode(ENB_PIN, OUTPUT);
  pinMode(IN3_PIN, OUTPUT);
  pinMode(IN4_PIN, OUTPUT);

  Serial.begin(9600);
  stopMotors();
  lastCommandTime = millis();
}

void loop() {
  pollSerial();

  unsigned long now = millis();

  // Safety: stop rover if Python command stream is lost.
  if (now - lastCommandTime > CMD_TIMEOUT_MS) {
    stopMotors();
  }

  // Periodically publish ultrasonic range.
  if (now - lastSensorTime >= SENSOR_PERIOD_MS) {
    long distanceCm = readDistanceCM();
    Serial.print("DIST:");
    Serial.println(distanceCm);
    lastSensorTime = now;
  }
}
