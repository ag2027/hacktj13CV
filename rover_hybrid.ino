/*
  Arduino Rover Controller (Standalone + Serial Manual)
  ------------------------------------------------------
  Board: Arduino Uno
  Sensor: HC-SR04 ultrasonic sensor
  Driver: L298N motor driver

  Works standalone in AUTO mode (default). Python is optional.

  Serial protocol:
    Outgoing:
      DIST:<cm>
      MODE:AUTO or MODE:MANUAL
      ACK:<command>
      ERR:<reason>

    Incoming:
      MODE:AUTO
      MODE:MANUAL
      FORWARD, LEFT, RIGHT, STOP
      FORWARD:<ms>, LEFT:<ms>, RIGHT:<ms>
      SET:STOP:<cm>
      SET:TURN:<cm>
      PING
*/

// ----------------------------
// Pin assignments
// ----------------------------
const int TRIG_PIN = 9;
const int ECHO_PIN = 10;

const int ENA_PIN = 5;   // PWM left motor enable
const int IN1_PIN = 2;
const int IN2_PIN = 3;

const int ENB_PIN = 6;   // PWM right motor enable
const int IN3_PIN = 7;
const int IN4_PIN = 8;

// ----------------------------
// Control parameters
// ----------------------------
const int DRIVE_SPEED = 170;
const unsigned long SENSOR_PERIOD_MS = 120;
const unsigned long DECISION_PERIOD_MS = 130;
const unsigned long CMD_TIMEOUT_MS = 1200;
const unsigned long MAX_TIMED_MOVE_MS = 3000;

// Autonomous behavior thresholds
long STOP_DISTANCE_CM = 12;
long TURN_DISTANCE_CM = 30;
unsigned long AUTO_TURN_MS = 280;

// ----------------------------
// Runtime state
// ----------------------------
enum ControlMode { MODE_AUTO, MODE_MANUAL };
ControlMode mode = MODE_AUTO;

unsigned long lastSensorTime = 0;
unsigned long lastDecisionTime = 0;
unsigned long lastCommandTime = 0;

unsigned long timedActionEndsAt = 0;
bool timedActionActive = false;
bool turnLeftNext = true;
long latestDistanceCm = 400;

char cmdBuffer[48];
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
  digitalWrite(IN1_PIN, LOW);
  digitalWrite(IN2_PIN, HIGH);
  digitalWrite(IN3_PIN, HIGH);
  digitalWrite(IN4_PIN, LOW);
}

void turnRight() {
  analogWrite(ENA_PIN, DRIVE_SPEED);
  analogWrite(ENB_PIN, DRIVE_SPEED);
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

// Required function name
long readDistanceCM() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  unsigned long duration = pulseIn(ECHO_PIN, HIGH, 25000UL);
  if (duration == 0) {
    return 400;
  }
  return (long)(duration * 0.0343 / 2.0);
}

void beginTimedAction(unsigned long durationMs) {
  if (durationMs > MAX_TIMED_MOVE_MS) {
    durationMs = MAX_TIMED_MOVE_MS;
  }
  timedActionEndsAt = millis() + durationMs;
  timedActionActive = true;
}

void clearTimedAction() {
  timedActionActive = false;
  timedActionEndsAt = 0;
}

void setMode(ControlMode newMode) {
  mode = newMode;
  clearTimedAction();
  stopMotors();
  if (mode == MODE_AUTO) {
    Serial.println("MODE:AUTO");
  } else {
    Serial.println("MODE:MANUAL");
  }
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

bool parseTimedCommand(const char* command, char* actionOut, unsigned long* durationOut) {
  const char* colon = strchr(command, ':');
  if (colon == NULL) return false;

  int actionLen = (int)(colon - command);
  if (actionLen <= 0 || actionLen > 10) return false;

  strncpy(actionOut, command, actionLen);
  actionOut[actionLen] = '\0';

  long parsed = atol(colon + 1);
  if (parsed <= 0) return false;

  *durationOut = (unsigned long)parsed;
  return true;
}

void processManualCommand(const char* command) {
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

  // Timed motion commands
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

  // Continuous movement commands
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

void processSystemCommand(const char* command) {
  if (strcmp(command, "MODE:AUTO") == 0) {
    setMode(MODE_AUTO);
    return;
  }

  if (strcmp(command, "MODE:MANUAL") == 0) {
    setMode(MODE_MANUAL);
    return;
  }

  if (strncmp(command, "SET:STOP:", 9) == 0) {
    long v = atol(command + 9);
    if (v >= 5 && v <= 100) {
      STOP_DISTANCE_CM = v;
      Serial.print("ACK:SET:STOP:");
      Serial.println(STOP_DISTANCE_CM);
    } else {
      Serial.println("ERR:SET:STOP");
    }
    return;
  }

  if (strncmp(command, "SET:TURN:", 9) == 0) {
    long v = atol(command + 9);
    if (v > STOP_DISTANCE_CM && v <= 200) {
      TURN_DISTANCE_CM = v;
      Serial.print("ACK:SET:TURN:");
      Serial.println(TURN_DISTANCE_CM);
    } else {
      Serial.println("ERR:SET:TURN");
    }
    return;
  }

  // Any remaining command is treated as manual motion command.
  processManualCommand(command);
}

void pollSerial() {
  while (Serial.available() > 0) {
    char c = (char)Serial.read();

    if (c == '\n' || c == '\r') {
      if (cmdIndex > 0) {
        cmdBuffer[cmdIndex] = '\0';
        processSystemCommand(cmdBuffer);
        cmdIndex = 0;
      }
    } else if (cmdIndex < sizeof(cmdBuffer) - 1) {
      cmdBuffer[cmdIndex++] = c;
    } else {
      cmdIndex = 0; // overflow guard
    }
  }
}

void autoDecideAndDrive() {
  // If timed action active (turn burst), let it complete.
  if (timedActionActive) {
    return;
  }

  if (latestDistanceCm <= STOP_DISTANCE_CM) {
    stopMotors();

    // In very close range, perform a short alternating turn to escape.
    if (turnLeftNext) {
      turnLeft();
    } else {
      turnRight();
    }
    turnLeftNext = !turnLeftNext;
    beginTimedAction(AUTO_TURN_MS);
    return;
  }

  if (latestDistanceCm <= TURN_DISTANCE_CM) {
    // Obstacle in caution zone: short corrective turn.
    if (turnLeftNext) {
      turnLeft();
    } else {
      turnRight();
    }
    turnLeftNext = !turnLeftNext;
    beginTimedAction(AUTO_TURN_MS / 2);
    return;
  }

  moveForward();
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
  Serial.println("MODE:AUTO");
}

void loop() {
  pollSerial();
  unsigned long now = millis();

  // Timed action completion (both AUTO and MANUAL modes)
  if (timedActionActive && now >= timedActionEndsAt) {
    stopMotors();
    clearTimedAction();
    Serial.println("DONE");
  }

  // Sensor publishing
  if (now - lastSensorTime >= SENSOR_PERIOD_MS) {
    latestDistanceCm = readDistanceCM();
    Serial.print("DIST:");
    Serial.println(latestDistanceCm);
    lastSensorTime = now;
  }

  if (mode == MODE_AUTO) {
    if (now - lastDecisionTime >= DECISION_PERIOD_MS) {
      autoDecideAndDrive();
      lastDecisionTime = now;
    }
  } else {
    // Manual mode safety timeout
    if ((now - lastCommandTime > CMD_TIMEOUT_MS) && !timedActionActive) {
      stopMotors();
    }
  }
}
