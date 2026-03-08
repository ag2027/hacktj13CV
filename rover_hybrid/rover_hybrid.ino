#include <Servo.h>

/*
  Rover Hybrid Controller (Sample-Hardware Pinout)
  ------------------------------------------------
  Ultrasonic: Echo=A4, Trig=A5
  Servo: D3
  Line sensors: Right=D10, Mid=D4, Left=D2 (active-low)
  Motor: ENA=5 ENB=6 IN1=7 IN2=8 IN3=9 IN4=11

  Incoming serial commands:
    MODE:AUTO | MODE:MANUAL
    STOP | FORWARD | LEFT | RIGHT | BACK
    FORWARD:<ms> | LEFT:<ms> | RIGHT:<ms> | BACK:<ms>
    SET:STOP:<cm> | SET:TURN:<cm>
    SET:FORWARD_MS:<ms> | SET:TURN_MS:<ms>
    SET:HEADING:N|E|S|W
    SET:BOUNDS:minX,maxX,minY,maxY
    SET:POS:x,y
    SET:LINE:0|1
    GET:STATE
    PING
*/

// ----------------------------
// 1) Hardware pins
// ----------------------------
const int PIN_ECHO = A4;
const int PIN_TRIG = A5;
const int PIN_SERVO = 3;

const int PIN_LT_R = 10;
const int PIN_LT_M = 4;
const int PIN_LT_L = 2;

const int ENA = 5;
const int ENB = 6;
const int IN1 = 7;
const int IN2 = 8;
const int IN3 = 9;
const int IN4 = 11;

Servo scanServo;

// ----------------------------
// 2) Control config
// ----------------------------
const int MOTOR_SPEED = 180;
const int FORWARD_LEFT_TRIM = 20;  // Left side slower by this amount to correct drift.
const unsigned long SENSOR_PERIOD_MS = 120;
const unsigned long DECISION_PERIOD_MS = 120;
const unsigned long CMD_TIMEOUT_MS = 1500;
const unsigned long MAX_TIMED_MOVE_MS = 3000;

const bool USE_LINE_SENSORS_DEFAULT = false;
const byte EDGE_HIT_CONFIRM_COUNT = 3;
const byte OBSTACLE_HIT_CONFIRM_COUNT = 2;

long STOP_DISTANCE_CM = 12;
long TURN_DISTANCE_CM = 30;
long ULTRA_AVOID_DISTANCE_CM = 10;

unsigned long FORWARD_STEP_MS = 500;
unsigned long BYPASS_FORWARD_MS = 620;  // Extra clearance distance during obstacle bypass.
unsigned long TURN_STEP_MS = 520;
unsigned long ESCAPE_TURN_MS = 432;

// ----------------------------
// 3) State
// ----------------------------
enum ControlMode { MODE_AUTO, MODE_MANUAL };
enum MotionAction { ACTION_STOP, ACTION_FORWARD, ACTION_BACK, ACTION_LEFT, ACTION_RIGHT };
enum Heading { HEADING_N, HEADING_E, HEADING_S, HEADING_W };

ControlMode mode = MODE_AUTO;
Heading heading = HEADING_E;

long gridX = 0;
long gridY = 0;
long boundMinX = 0;
long boundMaxX = 6;
long boundMinY = 0;
long boundMaxY = 1;

bool turnLeftNext = true;
bool useLineSensors = USE_LINE_SENSORS_DEFAULT;
byte edgeHitStreak = 0;
byte obstacleHitStreak = 0;
bool snakeModeEnabled = true;
bool snakeMovingEast = true;
int snakeRowStep = 1;
bool snakeFlipAfterForward = false;
bool snakeFinished = false;
bool ultraBypassActive = false;
bool ultraBypassRightSide = true;
byte ultraBypassStep = 0;

unsigned long lastSensorTime = 0;
unsigned long lastDecisionTime = 0;
unsigned long lastCommandTime = 0;

unsigned long timedActionEndsAt = 0;
bool timedActionActive = false;
MotionAction activeAction = ACTION_STOP;

long latestDistanceCm = 400;

char cmdBuffer[180];
byte cmdIndex = 0;

// ----------------------------
// 4) Motion helpers
// ----------------------------
void forward(uint8_t speedVal) {
  int leftSpeed = speedVal - FORWARD_LEFT_TRIM;
  if (leftSpeed < 0) leftSpeed = 0;
  analogWrite(ENA, leftSpeed);
  analogWrite(ENB, speedVal);
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, HIGH);
}

void back(uint8_t speedVal) {
  analogWrite(ENA, speedVal);
  analogWrite(ENB, speedVal);
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH);
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);
}

void left(uint8_t speedVal) {
  analogWrite(ENA, speedVal);
  analogWrite(ENB, speedVal);
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, HIGH);
}

void right(uint8_t speedVal) {
  analogWrite(ENA, speedVal);
  analogWrite(ENB, speedVal);
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);
}

void stopMotors() {
  analogWrite(ENA, 0);
  analogWrite(ENB, 0);
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);
}

void runAction(MotionAction action) {
  if (action == ACTION_FORWARD) forward(MOTOR_SPEED);
  else if (action == ACTION_BACK) back(MOTOR_SPEED);
  else if (action == ACTION_LEFT) left(MOTOR_SPEED);
  else if (action == ACTION_RIGHT) right(MOTOR_SPEED);
  else stopMotors();

  activeAction = action;
}

// ----------------------------
// 5) Sensors
// ----------------------------
int lineRight() { return digitalRead(PIN_LT_R) == LOW; }
int lineMid() { return digitalRead(PIN_LT_M) == LOW; }
int lineLeft() { return digitalRead(PIN_LT_L) == LOW; }

long readDistanceCM() {
  digitalWrite(PIN_TRIG, LOW);
  delayMicroseconds(2);
  digitalWrite(PIN_TRIG, HIGH);
  delayMicroseconds(20);
  digitalWrite(PIN_TRIG, LOW);

  unsigned long duration = pulseIn(PIN_ECHO, HIGH, 25000UL);
  if (duration == 0) return 400;

  long cm = (long)(duration / 58.0);
  if (cm < 3) return 400;  // Ignore invalid near-zero spikes
  return cm;
}

// ----------------------------
// 6) Pose / bounds
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

bool inBounds(long x, long y) {
  return x >= boundMinX && x <= boundMaxX && y >= boundMinY && y <= boundMaxY;
}

void forwardDelta(Heading h, int* dxOut, int* dyOut) {
  if (h == HEADING_N) {
    *dxOut = 0; *dyOut = -1;
  } else if (h == HEADING_E) {
    *dxOut = 1; *dyOut = 0;
  } else if (h == HEADING_S) {
    *dxOut = 0; *dyOut = 1;
  } else {
    *dxOut = -1; *dyOut = 0;
  }
}

bool forwardWouldLeaveBounds() {
  int dx = 0, dy = 0;
  forwardDelta(heading, &dx, &dy);
  return !inBounds(gridX + dx, gridY + dy);
}

void applyCompletedMotion(MotionAction action) {
  if (action == ACTION_LEFT) {
    heading = leftOf(heading);
    return;
  }
  if (action == ACTION_RIGHT) {
    heading = rightOf(heading);
    return;
  }
  if (action == ACTION_FORWARD) {
    int dx = 0, dy = 0;
    forwardDelta(heading, &dx, &dy);
    long nx = gridX + dx;
    long ny = gridY + dy;
    if (inBounds(nx, ny)) {
      gridX = nx;
      gridY = ny;
    }
    if (snakeFlipAfterForward) {
      snakeMovingEast = !snakeMovingEast;
      snakeFlipAfterForward = false;
    }
  }
}

bool edgeDetectedStable() {
  bool rawEdge = forwardWouldLeaveBounds();
  if (useLineSensors) rawEdge = rawEdge || lineLeft() || lineMid() || lineRight();

  if (rawEdge) {
    if (edgeHitStreak < 255) edgeHitStreak++;
  } else {
    edgeHitStreak = 0;
  }
  return edgeHitStreak >= EDGE_HIT_CONFIRM_COUNT;
}

bool obstacleDetectedStable() {
  bool nearObstacle = latestDistanceCm <= TURN_DISTANCE_CM;
  if (nearObstacle) {
    if (obstacleHitStreak < 255) obstacleHitStreak++;
  } else {
    obstacleHitStreak = 0;
  }
  return obstacleHitStreak >= OBSTACLE_HIT_CONFIRM_COUNT;
}

MotionAction bypassActionForStep(byte step, bool rightSide) {
  // Right-side bypass:
  // RIGHT, FORWARD, LEFT, FORWARD, LEFT, FORWARD, RIGHT
  // Left-side bypass is mirrored.
  if (rightSide) {
    if (step == 0) return ACTION_RIGHT;
    if (step == 1) return ACTION_FORWARD;
    if (step == 2) return ACTION_LEFT;
    if (step == 3) return ACTION_FORWARD;
    if (step == 4) return ACTION_LEFT;
    if (step == 5) return ACTION_FORWARD;
    return ACTION_RIGHT;
  }

  if (step == 0) return ACTION_LEFT;
  if (step == 1) return ACTION_FORWARD;
  if (step == 2) return ACTION_RIGHT;
  if (step == 3) return ACTION_FORWARD;
  if (step == 4) return ACTION_RIGHT;
  if (step == 5) return ACTION_FORWARD;
  return ACTION_LEFT;
}

bool simulateActionStep(long x, long y, Heading h, MotionAction action, long* xOut, long* yOut, Heading* hOut) {
  long nx = x;
  long ny = y;
  Heading nh = h;

  if (action == ACTION_LEFT) {
    nh = leftOf(h);
  } else if (action == ACTION_RIGHT) {
    nh = rightOf(h);
  } else if (action == ACTION_FORWARD) {
    int dx = 0;
    int dy = 0;
    forwardDelta(h, &dx, &dy);
    nx = x + dx;
    ny = y + dy;
    if (!inBounds(nx, ny)) {
      return false;
    }
  }

  *xOut = nx;
  *yOut = ny;
  *hOut = nh;
  return true;
}

bool canExecuteBypass(bool rightSide) {
  long x = gridX;
  long y = gridY;
  Heading h = heading;

  for (byte i = 0; i < 7; i++) {
    MotionAction a = bypassActionForStep(i, rightSide);
    if (!simulateActionStep(x, y, h, a, &x, &y, &h)) {
      return false;
    }
  }

  return true;
}

unsigned long bypassDurationForStep(byte step, MotionAction action) {
  if (action != ACTION_FORWARD) return TURN_STEP_MS;
  // Sequence: R, F, L, F, L, F, R
  // Make the first bypass forward (right/left then forward) longer to avoid clipping.
  if (step == 1) return (BYPASS_FORWARD_MS * 3UL) / 2UL;
  // Keep the middle forward leg (step 3) longer for cleaner front clearance.
  if (step == 3) return (BYPASS_FORWARD_MS * 3UL) / 2UL;
  return BYPASS_FORWARD_MS;
}


// ----------------------------
// 7) Timed actions
// ----------------------------
void startTimedAction(MotionAction action, unsigned long ms) {
  if (ms > MAX_TIMED_MOVE_MS) ms = MAX_TIMED_MOVE_MS;
  runAction(action);
  timedActionEndsAt = millis() + ms;
  timedActionActive = true;
}

void stopAll() {
  stopMotors();
  timedActionActive = false;
  timedActionEndsAt = 0;
  activeAction = ACTION_STOP;
}

void serviceTimedAction(unsigned long now) {
  if (timedActionActive && now >= timedActionEndsAt) {
    applyCompletedMotion(activeAction);
    stopAll();
    Serial.println("DONE");
  }
}

// ----------------------------
// 8) Serial parsing
// ----------------------------
bool parseHeading(char c, Heading* out) {
  if (c == 'N') *out = HEADING_N;
  else if (c == 'E') *out = HEADING_E;
  else if (c == 'S') *out = HEADING_S;
  else if (c == 'W') *out = HEADING_W;
  else return false;
  return true;
}

bool parseTimed(const char* cmd, MotionAction* actionOut, unsigned long* msOut) {
  const char* colon = strchr(cmd, ':');
  if (colon == NULL) return false;

  int n = (int)(colon - cmd);
  if (n <= 0 || n > 10) return false;

  char name[12];
  strncpy(name, cmd, n);
  name[n] = '\0';

  long ms = atol(colon + 1);
  if (ms <= 0) return false;

  if (strcmp(name, "FORWARD") == 0) *actionOut = ACTION_FORWARD;
  else if (strcmp(name, "BACK") == 0) *actionOut = ACTION_BACK;
  else if (strcmp(name, "LEFT") == 0) *actionOut = ACTION_LEFT;
  else if (strcmp(name, "RIGHT") == 0) *actionOut = ACTION_RIGHT;
  else return false;

  *msOut = (unsigned long)ms;
  return true;
}

bool parseBounds(char* spec, long* minX, long* maxX, long* minY, long* maxY) {
  long v[4];
  byte i = 0;
  char* save = NULL;
  char* tok = strtok_r(spec, ",", &save);
  while (tok != NULL && i < 4) {
    v[i++] = atol(tok);
    tok = strtok_r(NULL, ",", &save);
  }
  if (i != 4 || tok != NULL) return false;
  if (v[0] > v[1] || v[2] > v[3]) return false;

  *minX = v[0]; *maxX = v[1]; *minY = v[2]; *maxY = v[3];
  return true;
}

bool parseXY(char* spec, long* x, long* y) {
  char* save = NULL;
  char* xTok = strtok_r(spec, ",", &save);
  char* yTok = strtok_r(NULL, ",", &save);
  if (xTok == NULL || yTok == NULL) return false;

  *x = atol(xTok);
  *y = atol(yTok);
  return true;
}

void setMode(ControlMode newMode) {
  mode = newMode;
  stopAll();
  Serial.println(mode == MODE_AUTO ? "MODE:AUTO" : "MODE:MANUAL");
}

void processMotionCommand(const char* cmd) {
  lastCommandTime = millis();

  if (strcmp(cmd, "STOP") == 0) {
    stopAll();
    Serial.println("ACK:STOP");
    return;
  }

  if (strcmp(cmd, "FORWARD") == 0 || strcmp(cmd, "BACK") == 0 || strcmp(cmd, "LEFT") == 0 || strcmp(cmd, "RIGHT") == 0) {
    MotionAction a = ACTION_STOP;
    if (strcmp(cmd, "FORWARD") == 0) a = ACTION_FORWARD;
    else if (strcmp(cmd, "BACK") == 0) a = ACTION_BACK;
    else if (strcmp(cmd, "LEFT") == 0) a = ACTION_LEFT;
    else a = ACTION_RIGHT;

    stopAll();
    runAction(a);
    Serial.print("ACK:");
    Serial.println(cmd);
    return;
  }

  MotionAction a;
  unsigned long ms;
  if (parseTimed(cmd, &a, &ms)) {
    stopAll();
    startTimedAction(a, ms);
    Serial.print("ACK:");
    Serial.print((a == ACTION_FORWARD) ? "FORWARD" : (a == ACTION_BACK) ? "BACK" : (a == ACTION_LEFT) ? "LEFT" : "RIGHT");
    Serial.print(":");
    Serial.println(ms);
    return;
  }

  Serial.print("ERR:UNKNOWN:");
  Serial.println(cmd);
}

void processCommand(char* cmd) {
  if (strcmp(cmd, "PING") == 0) {
    Serial.println("PONG");
    return;
  }

  if (strcmp(cmd, "MODE:AUTO") == 0) {
    setMode(MODE_AUTO);
    return;
  }
  if (strcmp(cmd, "MODE:MANUAL") == 0) {
    setMode(MODE_MANUAL);
    return;
  }

  if (strncmp(cmd, "SET:STOP:", 9) == 0) {
    long v = atol(cmd + 9);
    if (v >= 5 && v <= 120) {
      STOP_DISTANCE_CM = v;
      Serial.print("ACK:SET:STOP:");
      Serial.println(STOP_DISTANCE_CM);
      obstacleHitStreak = 0;
    } else {
      Serial.println("ERR:SET:STOP");
    }
    return;
  }

  if (strncmp(cmd, "SET:TURN:", 9) == 0) {
    long v = atol(cmd + 9);
    if (v > STOP_DISTANCE_CM && v <= 250) {
      TURN_DISTANCE_CM = v;
      Serial.print("ACK:SET:TURN:");
      Serial.println(TURN_DISTANCE_CM);
      obstacleHitStreak = 0;
    } else {
      Serial.println("ERR:SET:TURN");
    }
    return;
  }

  if (strncmp(cmd, "SET:FORWARD_MS:", 15) == 0) {
    long v = atol(cmd + 15);
    if (v >= 80 && v <= (long)MAX_TIMED_MOVE_MS) {
      FORWARD_STEP_MS = (unsigned long)v;
      Serial.print("ACK:SET:FORWARD_MS:");
      Serial.println(FORWARD_STEP_MS);
    } else {
      Serial.println("ERR:SET:FORWARD_MS");
    }
    return;
  }

  if (strncmp(cmd, "SET:TURN_MS:", 12) == 0) {
    long v = atol(cmd + 12);
    if (v >= 80 && v <= (long)MAX_TIMED_MOVE_MS) {
      TURN_STEP_MS = (unsigned long)v;
      Serial.print("ACK:SET:TURN_MS:");
      Serial.println(TURN_STEP_MS);
    } else {
      Serial.println("ERR:SET:TURN_MS");
    }
    return;
  }

  if (strncmp(cmd, "SET:HEADING:", 12) == 0) {
    Heading h;
    if (parseHeading(cmd[12], &h)) {
      heading = h;
      Serial.print("ACK:SET:HEADING:");
      Serial.println(headingName(heading));
    } else {
      Serial.println("ERR:SET:HEADING");
    }
    return;
  }

  if (strncmp(cmd, "SET:BOUNDS:", 11) == 0) {
    long minX, maxX, minY, maxY;
    if (parseBounds(cmd + 11, &minX, &maxX, &minY, &maxY)) {
      boundMinX = minX;
      boundMaxX = maxX;
      boundMinY = minY;
      boundMaxY = maxY;
      if (!inBounds(gridX, gridY)) {
        gridX = boundMinX;
        gridY = boundMinY;
      }
      edgeHitStreak = 0;
      Serial.print("ACK:SET:BOUNDS:");
      Serial.print(boundMinX); Serial.print(",");
      Serial.print(boundMaxX); Serial.print(",");
      Serial.print(boundMinY); Serial.print(",");
      Serial.println(boundMaxY);
    } else {
      Serial.println("ERR:SET:BOUNDS");
    }
    return;
  }

  if (strncmp(cmd, "SET:POS:", 8) == 0) {
    long x, y;
    if (parseXY(cmd + 8, &x, &y) && inBounds(x, y)) {
      gridX = x;
      gridY = y;
      edgeHitStreak = 0;
      Serial.print("ACK:SET:POS:");
      Serial.print(gridX); Serial.print(",");
      Serial.println(gridY);
    } else {
      Serial.println("ERR:SET:POS");
    }
    return;
  }

  if (strncmp(cmd, "SET:LINE:", 9) == 0) {
    int v = atoi(cmd + 9);
    if (v == 0 || v == 1) {
      useLineSensors = (v == 1);
      edgeHitStreak = 0;
      Serial.print("ACK:SET:LINE:");
      Serial.println(useLineSensors ? 1 : 0);
    } else {
      Serial.println("ERR:SET:LINE");
    }
    return;
  }

  if (strncmp(cmd, "SET:SNAKE:", 10) == 0) {
    int v = atoi(cmd + 10);
    if (v == 0 || v == 1) {
      snakeModeEnabled = (v == 1);
      snakeFinished = false;
      snakeFlipAfterForward = false;
      Serial.print("ACK:SET:SNAKE:");
      Serial.println(snakeModeEnabled ? 1 : 0);
    } else {
      Serial.println("ERR:SET:SNAKE");
    }
    return;
  }

  if (strcmp(cmd, "GET:STATE") == 0) {
    Serial.print("STATE:POS:");
    Serial.print(gridX); Serial.print(",");
    Serial.print(gridY); Serial.print(";HEADING:");
    Serial.print(headingName(heading));
    Serial.print(";BOUNDS:");
    Serial.print(boundMinX); Serial.print(",");
    Serial.print(boundMaxX); Serial.print(",");
    Serial.print(boundMinY); Serial.print(",");
    Serial.print(boundMaxY);
    Serial.print(";LINE:");
    Serial.print(useLineSensors ? 1 : 0);
    Serial.print(";SNAKE:");
    Serial.print(snakeModeEnabled ? 1 : 0);
    Serial.print(";DIR:");
    Serial.println(snakeMovingEast ? "EAST" : "WEST");
    return;
  }

  processMotionCommand(cmd);
}

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
      cmdIndex = 0;
      Serial.println("ERR:CMD_OVERFLOW");
    }
  }
}

// ----------------------------
// 9) Autonomous policy
// ----------------------------
void autoDecideAndDrive() {
  if (timedActionActive) return;

  if (snakeModeEnabled) {
    if (snakeFinished) {
      stopAll();
      return;
    }

    // Continue a multi-step bypass if already active.
    if (ultraBypassActive) {
      MotionAction nextBypassAction = bypassActionForStep(ultraBypassStep, ultraBypassRightSide);
      unsigned long dur = bypassDurationForStep(ultraBypassStep, nextBypassAction);
      startTimedAction(nextBypassAction, dur);
      ultraBypassStep++;
      if (ultraBypassStep >= 7) {
        ultraBypassActive = false;
        ultraBypassStep = 0;
      }
      return;
    }

    // On ultrasonic trigger, start bypass sequence around the obstacle.
    if (latestDistanceCm <= ULTRA_AVOID_DISTANCE_CM) {
      bool rightOK = canExecuteBypass(true);
      bool leftOK = canExecuteBypass(false);

      if (rightOK || leftOK) {
        ultraBypassRightSide = rightOK ? true : false;
        ultraBypassActive = true;
        ultraBypassStep = 0;

        MotionAction firstAction = bypassActionForStep(ultraBypassStep, ultraBypassRightSide);
        unsigned long firstDur = bypassDurationForStep(ultraBypassStep, firstAction);
        startTimedAction(firstAction, firstDur);
        ultraBypassStep++;
      } else {
        // Tight corner fallback.
        startTimedAction(ACTION_RIGHT, TURN_STEP_MS);
      }
      return;
    }
    Heading rowHeading = snakeMovingEast ? HEADING_E : HEADING_W;
    Heading rowStepHeading = (snakeRowStep > 0) ? HEADING_S : HEADING_N;
    bool atRowEdge = snakeMovingEast ? (gridX >= boundMaxX) : (gridX <= boundMinX);

    if (!atRowEdge) {
      if (heading != rowHeading) {
        MotionAction turnAction = (rightOf(heading) == rowHeading) ? ACTION_RIGHT : ACTION_LEFT;
        startTimedAction(turnAction, TURN_STEP_MS);
      } else {
        startTimedAction(ACTION_FORWARD, FORWARD_STEP_MS);
      }
      return;
    }

    bool canStepRow = (snakeRowStep > 0) ? (gridY < boundMaxY) : (gridY > boundMinY);
    if (!canStepRow) {
      snakeFinished = true;
      stopAll();
      Serial.println("ACK:SNAKE:DONE");
      return;
    }

    if (heading != rowStepHeading) {
      MotionAction turnAction = (rightOf(heading) == rowStepHeading) ? ACTION_RIGHT : ACTION_LEFT;
      startTimedAction(turnAction, TURN_STEP_MS);
    } else {
      snakeFlipAfterForward = true;
      startTimedAction(ACTION_FORWARD, FORWARD_STEP_MS);
    }
    return;
  }

  // Fallback obstacle-aware mode when snake is disabled.
  if (edgeDetectedStable()) {
    MotionAction turnAction = turnLeftNext ? ACTION_LEFT : ACTION_RIGHT;
    turnLeftNext = !turnLeftNext;
    startTimedAction(turnAction, TURN_STEP_MS);
    return;
  }

  if (latestDistanceCm <= STOP_DISTANCE_CM) {
    MotionAction turnAction = turnLeftNext ? ACTION_LEFT : ACTION_RIGHT;
    turnLeftNext = !turnLeftNext;
    startTimedAction(turnAction, ESCAPE_TURN_MS);
    return;
  }

  if (obstacleDetectedStable()) {
    MotionAction turnAction = turnLeftNext ? ACTION_LEFT : ACTION_RIGHT;
    turnLeftNext = !turnLeftNext;
    startTimedAction(turnAction, TURN_STEP_MS);
    return;
  }

  startTimedAction(ACTION_FORWARD, FORWARD_STEP_MS);
}

// ----------------------------
// 10) Arduino lifecycle
// ----------------------------
void setup() {
  scanServo.attach(PIN_SERVO, 700, 2400);
  scanServo.write(90);

  Serial.begin(9600);

  pinMode(PIN_ECHO, INPUT);
  pinMode(PIN_TRIG, OUTPUT);

  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);
  pinMode(ENA, OUTPUT);
  pinMode(ENB, OUTPUT);

  // Active-low sensors are usually stable with pullups.
  pinMode(PIN_LT_R, INPUT_PULLUP);
  pinMode(PIN_LT_M, INPUT_PULLUP);
  pinMode(PIN_LT_L, INPUT_PULLUP);

  stopAll();
  lastCommandTime = millis();

  Serial.println("READY");
  Serial.println("MODE:AUTO");
}

void loop() {
  pollSerial();

  unsigned long now = millis();
  serviceTimedAction(now);

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
    if ((now - lastCommandTime > CMD_TIMEOUT_MS) && !timedActionActive) {
      stopAll();
    }
  }
}










