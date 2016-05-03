#pragma once

#include "inputprocessing.h"

using namespace cv;
using namespace std;

const String WINDOW_NAME = "testingWindow";

// actual number of rows will be POINT_ROWS + 1, same with POINT_COLS,
// to show points on bottom and  right sides
const int TRAINING_POINT_ROWS  = 3;
const int TRAINING_POINT_COLS = 4;
const int TRAINING_FRAME_COUNT = 12;

bool trainingPhase(InputProcessing ip, Size screenResolution, const char * trainingFile);
int getTrainingCount();