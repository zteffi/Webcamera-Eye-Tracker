#pragma once

#include "inputprocessing.h"

using namespace cv;
using namespace std;

const String WINDOW_NAME = "testingWindow";

// actual number of rows will be POINT_ROWS + 1, same with POINT_COLS,
// to show points on bottom and  right sides
const int POINT_ROWS  = 2;
const int POINT_COLS = 3;
const int FRAME_COUNT = 4;

void trainingPhase(InputProcessing ip, Size screenResolution);