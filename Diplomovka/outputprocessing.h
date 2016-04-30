#pragma once
#include "inputprocessing.h"


struct outdata{
	Point rightOuterCorner;
	Point rightCenter;
	Point rightInnerCorner;
	Point leftInnerCorner;
	Point leftCenter;
	Point leftOuterCorner;

	Point screenX;
	Point screenY;
};

class OutputProcessing {

public:

	outdata [] trainData;
	outdata [] testData;
};