/*
  Arduino Rover Controller (Hybrid: Sensor Auto + Path Execution)
  ---------------------------------------------------------------
  Board: Arduino Uno
  Sensor: HC-SR04 ultrasonic sensor
  Driver: L298N motor driver

  Purpose:
  - Mirror the Python/QML control flow directly on Arduino:
    1) Distance-gated autonomous policy (STOP / TURN / FORWARD)
    2) Timed motor action executor (FORWARD:<ms>, LEFT:<ms>, RIGHT:<ms>)
    3) Grid path -> turn/forward conversion (like patrol_to_arduino.py)

  Serial protocol:
    Outgoing:
      READY
      DIST:<cm>
      MODE:AUTO or MODE:MANUAL
      ACK:<...>
      DONE
      ERR:<reason>

    Incoming:
      MODE:AUTO
      MODE:MANUAL
      STOP
      FORWARD, LEFT, RIGHT
      FORWARD:<ms>, LEFT:<ms>, RIGHT:<ms>
      PATH:x1,y1|x2,y2|x3,y3|...
      QTURN:<hint_0_to_100>
      SET:STOP:<cm>
      SET:TURN:<cm>
      SET:FORWARD_MS:<ms>
      SET:TURN_MS:<ms>
      SET:HEADING:N|E|S|W
      PING
*/

#include <math.h>
#include <stdlib.h>
#include <string.h>

// ----------------------------
// 1) Pin assignments
// ----------------------------
const int TRIG_PIN = 9;
const int ECHO_PIN = 10;

const int ENA_PIN = 5;
const int IN1_PIN = 2;
const int IN2_PIN = 3;

const int ENB_PIN = 6;
const int IN3_PIN = 7;
const int IN4_PIN = 8;

// ----------------------------
// 2) Timing + policy parameters
// ----------------------------
const int DRIVE_SPEED = 170;
const unsigned long SENSOR_PERIOD_MS = 120;
const unsigned long DECISION_PERIOD_MS = 130;
const unsigned long CMD_TIMEOUT_MS = 1200;
const unsigned long MAX_TIMED_MOVE_MS = 3000;

long STOP_DISTANCE_CM = 12;
long TURN_DISTANCE_CM = 30;

unsigned long FORWARD_STEP_MS = 450;
unsigned long TURN_STEP_MS = 330;
unsigned long AUTO_ESCAPE_TURN_MS = 280;
unsigned long AUTO_CAUTION_TURN_MS = 140;

// ----------------------------
// 3) Runtime state
// ----------------------------
enum ControlMode { MODE_AUTO, MODE_MANUAL };
enum MotionAction { ACTION_STOP, ACTION_FORWARD, ACTION_LEFT, ACTION_RIGHT };
enum Heading { HEADING_N, HEADING_E, HEADING_S, HEADING_W };

ControlMode mode = MODE_AUTO;
Heading roverHeading = HEADING_E;

unsigned long lastSensorTime = 0;
unsigned long lastDecisionTime = 0;
unsigned long lastCommandTime = 0;

unsigned long timedActionEndsAt = 0;
bool timedActionActive = false;
MotionAction activeAction = ACTION_STOP;

long latestDistanceCm = 400;
bool turnLeftNext = true;

char cmdBuffer[240];
byte cmdIndex = 0;

// ----------------------------
// 4) Timed action queue
// ----------------------------
struct TimedAction {
  MotionAction action;
  unsigned long durationMs;
};

const byte ACTION_QUEUE_SIZE = 40;
TimedAction actionQueue[ACTION_QUEUE_SIZE];
byte actionQueueHead = 0;
byte actionQueueTail = 0;

bool queueIsEmpty() {
  return actionQueueHead == actionQueueTail;
}

bool queueIsFull() {
  return (byte)(actionQueueTail + 1) % ACTION_QUEUE_SIZE == actionQueueHead;
}

bool enqueueAction(MotionAction action, unsigned long durationMs) {
  if (queueIsFull()) return false;
  if (durationMs > MAX_TIMED_MOVE_MS) durationMs = MAX_TIMED_MOVE_MS;
  actionQueue[actionQueueTail].action = action;
  actionQueue[actionQueueTail].durationMs = durationMs;
  actionQueueTail = (byte)(actionQueueTail + 1) % ACTION_QUEUE_SIZE;
  return true;
}

bool dequeueAction(TimedAction* out) {
  if (queueIsEmpty()) return false;
  *out = actionQueue[actionQueueHead];
  actionQueueHead = (byte)(actionQueueHead + 1) % ACTION_QUEUE_SIZE;
  return true;
}

void clearQueue() {
  actionQueueHead = 0;
  actionQueueTail = 0;
}

// ----------------------------
// 5) Motor primitives
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

void runAction(MotionAction action) {
  if (action == ACTION_FORWARD) {
    moveForward();
  } else if (action == ACTION_LEFT) {
    turnLeft();
  } else if (action == ACTION_RIGHT) {
    turnRight();
  } else {
    stopMotors();
  }
  activeAction = action;
}

// ----------------------------
// 6) Sensor read
// ----------------------------
long readDistanceCM() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  unsigned long duration = pulseIn(ECHO_PIN, HIGH, 25000UL);
  if (duration == 0) return 400;
  return (long)(duration * 0.0343 / 2.0);
}

// ----------------------------
// 7) QML-inspired steering policy
// ----------------------------
float qmlInspiredRightProbability(long distanceCm, float turnHint01) {
  float d = (float)distanceCm;
  if (d < 0.0f) d = 0.0f;
  if (d > 100.0f) d = 100.0f;
  float dNorm = d / 100.0f;

  if (turnHint01 < 0.0f) turnHint01 = 0.0f;
  if (turnHint01 > 1.0f) turnHint01 = 1.0f;

  // Lightweight approximation of a learned score -> probability.
  float z = -0.55f + (1.90f * turnHint01) - (0.95f * dNorm);
  float p = 1.0f / (1.0f + expf(-3.0f * z));
  if (p < 0.0f) p = 0.0f;
  if (p > 1.0f) p = 1.0f;
  return p;
}

MotionAction decideAutoAction(long distanceCm, float turnHint01) {
  if (distanceCm <= STOP_DISTANCE_CM) return ACTION_STOP;
  if (distanceCm > TURN_DISTANCE_CM) return ACTION_FORWARD;

  float pRight = qmlInspiredRightProbability(distanceCm, turnHint01);
  return (pRight >= 0.5f) ? ACTION_RIGHT : ACTION_LEFT;
}

// ----------------------------
// 8) Heading + path conversion
// ----------------------------
const char* headingName(Heading h) {
  if (h == HEADING_N) return "N";
  if (h == HEADING_E) return "E";
  if (h == HEADING_S) return "S";
  return "W";
}

Heading rightOf(Heading h) {
  if (h == HEADING_N) return HEADING_E;
  if (h == HEADING_E) return HEADING_S;
  if (h == HEADING_S) return HEADING_W;
  return HEADING_N;
}

Heading leftOf(Heading h) {
  if (h == HEADING_N) return HEADING_W;
  if (h == HEADING_W) return HEADING_S;
  if (h == HEADING_S) return HEADING_E;
  return HEADING_N;
}

bool parsePointToken(const char* token, int* xOut, int* yOut) {
  const char* comma = strchr(token, ',');
  if (comma == NULL) return false;
  int leftLen = (int)(comma - token);
  if (leftLen <= 0 || leftLen > 10) return false;

  char xBuf[12];
  strncpy(xBuf, token, leftLen);
  xBuf[leftLen] = '\0';

  const char* yPart = comma + 1;
  if (strlen(yPart) == 0 || strlen(yPart) > 10) return false;

  *xOut = atoi(xBuf);
  *yOut = atoi(yPart);
  return true;
}

bool enqueueTurnSequence(Heading* h, Heading target) {
  if (*h == target) return true;
  if (rightOf(*h) == target) {
    if (!enqueueAction(ACTION_RIGHT, TURN_STEP_MS)) return false;
    *h = target;
    return true;
  }
  if (leftOf(*h) == target) {
    if (!enqueueAction(ACTION_LEFT, TURN_STEP_MS)) return false;
    *h = target;
    return true;
  }

  Heading first = rightOf(*h);
  if (!enqueueAction(ACTION_RIGHT, TURN_STEP_MS)) return false;
  if (!enqueueAction(ACTION_RIGHT, TURN_STEP_MS)) return false;
  *h = rightOf(first);
  return true;
}

bool enqueuePathMotion(char* spec) {
  int xPrev = 0, yPrev = 0;
  int xCur = 0, yCur = 0;
  bool first = true;
  Heading h = roverHeading;

  char* save = NULL;
  char* token = strtok_r(spec, "|", &save);
  while (token != NULL) {
    if (!parsePointToken(token, &xCur, &yCur)) return false;

    if (first) {
      xPrev = xCur;
      yPrev = yCur;
      first = false;
    } else {
      int dx = xCur - xPrev;
      int dy = yCur - yPrev;
      Heading target;

      if (dx == 1 && dy == 0) target = HEADING_E;
      else if (dx == -1 && dy == 0) target = HEADING_W;
      else if (dx == 0 && dy == 1) target = HEADING_S;
      else if (dx == 0 && dy == -1) target = HEADING_N;
      else return false;

      if (!enqueueTurnSequence(&h, target)) return false;
      if (!enqueueAction(ACTION_FORWARD, FORWARD_STEP_MS)) return false;

      xPrev = xCur;
      yPrev = yCur;
    }

    token = strtok_r(NULL, "|", &save);
  }

  roverHeading = h;
  return !first;
}

// ----------------------------
// 9) Timed action lifecycle
// ----------------------------
void startTimedAction(MotionAction action, unsigned long durationMs) {
  if (durationMs > MAX_TIMED_MOVE_MS) durationMs = MAX_TIMED_MOVE_MS;
  runAction(action);
  timedActionEndsAt = millis() + durationMs;
  timedActionActive = true;
}

void stopAllMotion() {
  stopMotors();
  timedActionActive = false;
  timedActionEndsAt = 0;
  activeAction = ACTION_STOP;
}

void serviceTimedActionQueue(unsigned long now) {
  if (timedActionActive && now >= timedActionEndsAt) {
    stopAllMotion();
    Serial.println("DONE");
  }

  if (!timedActionActive && !queueIsEmpty()) {
    TimedAction next;
    if (dequeueAction(&next)) {
      startTimedAction(next.action, next.durationMs);
      Serial.print("ACK:RUN:");
      Serial.print((next.action == ACTION_FORWARD) ? "FORWARD" : (next.action == ACTION_LEFT) ? "LEFT" : "RIGHT");
      Serial.print(":");
      Serial.println(next.durationMs);
    }
  }
}

// ----------------------------
// 10) Command parsing
// ----------------------------
bool parseTimedCommand(const char* command, MotionAction* actionOut, unsigned long* durationOut) {
  const char* colon = strchr(command, ':');
  if (colon == NULL) return false;

  int actionLen = (int)(colon - command);
  if (actionLen <= 0 || actionLen > 10) return false;

  char actionName[12];
  strncpy(actionName, command, actionLen);
  actionName[actionLen] = '\0';

  long parsed = atol(colon + 1);
  if (parsed <= 0) return false;

  if (strcmp(actionName, "FORWARD") == 0) *actionOut = ACTION_FORWARD;
  else if (strcmp(actionName, "LEFT") == 0) *actionOut = ACTION_LEFT;
  else if (strcmp(actionName, "RIGHT") == 0) *actionOut = ACTION_RIGHT;
  else return false;

  *durationOut = (unsigned long)parsed;
  return true;
}

void setMode(ControlMode newMode) {
  mode = newMode;
  clearQueue();
  stopAllMotion();
  Serial.println(mode == MODE_AUTO ? "MODE:AUTO" : "MODE:MANUAL");
}

void processManualMotionCommand(const char* command) {
  lastCommandTime = millis();

  if (strcmp(command, "STOP") == 0) {
    clearQueue();
    stopAllMotion();
    Serial.println("ACK:STOP");
    return;
  }

  if (strcmp(command, "FORWARD") == 0 || strcmp(command, "LEFT") == 0 || strcmp(command, "RIGHT") == 0) {
    MotionAction a = (strcmp(command, "FORWARD") == 0) ? ACTION_FORWARD : (strcmp(command, "LEFT") == 0) ? ACTION_LEFT : ACTION_RIGHT;
    clearQueue();
    stopAllMotion();
    runAction(a);
    Serial.print("ACK:");
    Serial.println(command);
    return;
  }

  MotionAction timedAction;
  unsigned long durationMs = 0;
  if (parseTimedCommand(command, &timedAction, &durationMs)) {
    clearQueue();
    stopAllMotion();
    startTimedAction(timedAction, durationMs);
    Serial.print("ACK:");
    Serial.print((timedAction == ACTION_FORWARD) ? "FORWARD" : (timedAction == ACTION_LEFT) ? "LEFT" : "RIGHT");
    Serial.print(":");
    Serial.println(durationMs);
    return;
  }

  Serial.print("ERR:UNKNOWN:");
  Serial.println(command);
}

bool parseHeadingChar(char c, Heading* out) {
  if (c == 'N') *out = HEADING_N;
  else if (c == 'E') *out = HEADING_E;
  else if (c == 'S') *out = HEADING_S;
  else if (c == 'W') *out = HEADING_W;
  else return false;
  return true;
}

void processSystemCommand(char* command) {
  if (strcmp(command, "PING") == 0) {
    Serial.println("PONG");
    return;
  }

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

  if (strncmp(command, "SET:FORWARD_MS:", 15) == 0) {
    long v = atol(command + 15);
    if (v >= 80 && v <= (long)MAX_TIMED_MOVE_MS) {
      FORWARD_STEP_MS = (unsigned long)v;
      Serial.print("ACK:SET:FORWARD_MS:");
      Serial.println(FORWARD_STEP_MS);
    } else {
      Serial.println("ERR:SET:FORWARD_MS");
    }
    return;
  }

  if (strncmp(command, "SET:TURN_MS:", 12) == 0) {
    long v = atol(command + 12);
    if (v >= 80 && v <= (long)MAX_TIMED_MOVE_MS) {
      TURN_STEP_MS = (unsigned long)v;
      Serial.print("ACK:SET:TURN_MS:");
      Serial.println(TURN_STEP_MS);
    } else {
      Serial.println("ERR:SET:TURN_MS");
    }
    return;
  }

  if (strncmp(command, "SET:HEADING:", 12) == 0) {
    Heading h;
    if (parseHeadingChar(command[12], &h)) {
      roverHeading = h;
      Serial.print("ACK:SET:HEADING:");
      Serial.println(headingName(roverHeading));
    } else {
      Serial.println("ERR:SET:HEADING");
    }
    return;
  }

  if (strncmp(command, "QTURN:", 6) == 0) {
    int hintInt = atoi(command + 6);
    if (hintInt < 0) hintInt = 0;
    if (hintInt > 100) hintInt = 100;
    float hint = ((float)hintInt) / 100.0f;

    MotionAction turnAction = decideAutoAction(latestDistanceCm, hint);
    if (turnAction == ACTION_LEFT || turnAction == ACTION_RIGHT) {
      clearQueue();
      stopAllMotion();
      startTimedAction(turnAction, TURN_STEP_MS);
      Serial.print("ACK:QTURN:");
      Serial.println((turnAction == ACTION_RIGHT) ? "RIGHT" : "LEFT");
    } else {
      Serial.println("ACK:QTURN:NONE");
    }
    return;
  }

  if (strncmp(command, "PATH:", 5) == 0) {
    clearQueue();
    stopAllMotion();
    bool ok = enqueuePathMotion(command + 5);
    if (ok) {
      Serial.print("ACK:PATH:QSIZE:");
      byte size = (actionQueueTail + ACTION_QUEUE_SIZE - actionQueueHead) % ACTION_QUEUE_SIZE;
      Serial.println(size);
    } else {
      clearQueue();
      Serial.println("ERR:PATH");
    }
    return;
  }

  // Default motion command handling.
  processManualMotionCommand(command);
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
      cmdIndex = 0;
      Serial.println("ERR:CMD_OVERFLOW");
    }
  }
}

// ----------------------------
// 11) Autonomous loop
// ----------------------------
void autoDecideAndDrive() {
  // Let queued/timed actions finish before issuing a new policy decision.
  if (timedActionActive || !queueIsEmpty()) return;

  MotionAction decision;
  if (latestDistanceCm <= STOP_DISTANCE_CM) {
    // Close obstacle: stronger turn burst, alternating direction.
    decision = turnLeftNext ? ACTION_LEFT : ACTION_RIGHT;
    turnLeftNext = !turnLeftNext;
    startTimedAction(decision, AUTO_ESCAPE_TURN_MS);
    return;
  }

  if (latestDistanceCm <= TURN_DISTANCE_CM) {
    // Caution zone: use QML-inspired gate with alternating hint.
    float hint = turnLeftNext ? 0.0f : 1.0f;
    decision = decideAutoAction(latestDistanceCm, hint);
    if (decision != ACTION_LEFT && decision != ACTION_RIGHT) {
      decision = turnLeftNext ? ACTION_LEFT : ACTION_RIGHT;
    }
    turnLeftNext = !turnLeftNext;
    startTimedAction(decision, AUTO_CAUTION_TURN_MS);
    return;
  }

  runAction(ACTION_FORWARD);
}

// ----------------------------
// 12) Arduino lifecycle
// ----------------------------
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
  stopAllMotion();
  lastCommandTime = millis();

  Serial.println("READY");
  Serial.println("MODE:AUTO");
}

void loop() {
  pollSerial();
  unsigned long now = millis();

  serviceTimedActionQueue(now);

  // Publish distance for telemetry/debug (same as Python contract: DIST:<cm>).
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
    // Safety timeout for manual mode when no queued/timed action is running.
    if ((now - lastCommandTime > CMD_TIMEOUT_MS) && !timedActionActive && queueIsEmpty()) {
      stopAllMotion();
    }
  }
}
