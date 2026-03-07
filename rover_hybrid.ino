/*
  Hybrid Rover Controller (Arduino Side)
  --------------------------------------
  Board: Arduino Uno
  Sensors: HC-SR04 ultrasonic sensor
  Driver: L298N motor driver

  Protocol:
    - Outgoing sensor line: DIST:<value_cm>
    - Incoming commands:
        FORWARD, LEFT, RIGHT, STOP
        FORWARD:<ms>, LEFT:<ms>, RIGHT:<ms>
        PING
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
const int DRIVE_SPEED = 170;                 // 0-255
const unsigned long SENSOR_PERIOD_MS = 120;  // Sensor publish rate
const unsigned long CMD_TIMEOUT_MS = 1200;   // Safety timeout
const unsigned long MAX_TIMED_MOVE_MS = 3000;

// ----------------------------
// Runtime state
// ----------------------------
unsigned long lastSensorTime = 0;
unsigned long lastCommandTime = 0;
unsigned long timedActionEndsAt = 0;
bool timedActionActive = false;

char cmdBuffer[40];
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

void beginTimedAction(unsigned long durationMs) {
  unsigned long clamped = durationMs;
  if (clamped > MAX_TIMED_MOVE_MS) {
    clamped = MAX_TIMED_MOVE_MS;
  }
  timedActionEndsAt = millis() + clamped;
  timedActionActive = true;
}

void clearTimedAction() {
  timedActionActive = false;
  timedActionEndsAt = 0;
}

bool parseTimedCommand(const char* command, char* actionOut, unsigned long* durationOut) {
  const char* colon = strchr(command, ':');
  if (colon == NULL) {
    return false;
  }

  int actionLen = (int)(colon - command);
  if (actionLen <= 0 || actionLen > 10) {
    return false;
  }

  strncpy(actionOut, command, actionLen);
  actionOut[actionLen] = '\0';

  long parsed = atol(colon + 1);
  if (parsed <= 0) {
    return false;
  }

  *durationOut = (unsigned long)parsed;
  return true;
}

void executeActionName(const char* action) {
  if (strcmp(action, "FORWARD") == 0) {
    moveForward();
  } else if (strcmp(action, "LEFT") == 0) {
    turnLeft();
  } else if (strcmp(action, "RIGHT") == 0) {
    turnRight();
  } else {
    stopMotors();
  }
}

void processCommand(const char* command) {
  lastCommandTime = millis();

  if (strcmp(command, "PING") == 0) {
    Serial.println("PONG");
    return;
  }

  if (strcmp(command, "STOP") == 0) {
    stopMotors();
    clearTimedAction();
    Serial.println("ACK:STOP");
    return;
  }

  // Timed commands: FORWARD:450 / LEFT:330 / RIGHT:330
  char actionName[12];
  unsigned long durationMs = 0;
  if (parseTimedCommand(command, actionName, &durationMs)) {
    if (strcmp(actionName, "FORWARD") == 0 || strcmp(actionName, "LEFT") == 0 || strcmp(actionName, "RIGHT") == 0) {
      executeActionName(actionName);
      beginTimedAction(durationMs);
      Serial.print("ACK:");
      Serial.print(actionName);
      Serial.print(":");
      Serial.println(durationMs);
      return;
    }
  }

  // Non-timed continuous commands
  if (strcmp(command, "FORWARD") == 0 || strcmp(command, "LEFT") == 0 || strcmp(command, "RIGHT") == 0) {
    executeActionName(command);
    clearTimedAction();
    Serial.print("ACK:");
    Serial.println(command);
    return;
  }

  Serial.print("ERR:UNKNOWN:");
  Serial.println(command);
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
      cmdIndex = 0; // overflow guard
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
  Serial.println("READY");
}

void loop() {
  pollSerial();

  unsigned long now = millis();

  // Stop after timed action duration elapses.
  if (timedActionActive && now >= timedActionEndsAt) {
    stopMotors();
    clearTimedAction();
    Serial.println("DONE");
  }

  // Safety: stop rover if Python command stream is lost.
  if ((now - lastCommandTime > CMD_TIMEOUT_MS) && !timedActionActive) {
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
