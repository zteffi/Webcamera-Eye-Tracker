#include <windows.h>

#include "inputprocessing.h"

Size getScreenRes(float scaleFactor = 1);
Mat  hwnd2mat(HWND hwnd, float scaleFactor);
/*
captures both screen and webcam input
*/
long captureVids(InputProcessing ip, string fileName, float scaleFactor);
/*
processes and removes video capture data
*/
void processTrackingData(InputProcessing ip, string folderName, long lastFrameNum, const char * trackFile);

/*
writes predictions into outputfile and draws predicitons onto screencapture images returns success
loadMatFromCSVFile(inputFile, Mat, 11, inputCout) returns 11 x inputCount matrix, last 2 cols are targets)
trackfile 9x trackcount matrix
outputfile 2x trackcount matrix
*/
bool processOutput(InputProcessing ip, long trainCount, const char * trainFile, long trackCount, const char * trackFile, 
	const char * outputFile, string outputImagesFolderName, Size screenSize);