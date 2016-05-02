
#include "tracking_phase.h"

HWND hDesktopWnd;

Size getScreenRes(float scaleFactor) {
	
	// get the height and width of the screen
	int height = GetSystemMetrics(SM_CYVIRTUALSCREEN);
	int width = GetSystemMetrics(SM_CXVIRTUALSCREEN);
	height = (int)(height * scaleFactor);
	width = (int)(width * scaleFactor);
	return Size(width, height);
}

Mat  hwnd2mat(HWND hwnd, float scaleFactor) {
	HDC hwindowDC, hwindowCompatibleDC;

	int height, width, srcheight, srcwidth;
	HBITMAP hbwindow;
	Mat src;
	BITMAPINFOHEADER  bi;

	hwindowDC = GetDC(hwnd);
	hwindowCompatibleDC = CreateCompatibleDC(hwindowDC);
	SetStretchBltMode(hwindowCompatibleDC, COLORONCOLOR);

	RECT windowsize;    // get the height and width of the screen
	GetClientRect(hwnd, &windowsize);

	srcheight = windowsize.bottom;
	srcwidth = windowsize.right;
	height = (int)(windowsize.bottom * scaleFactor);  //change this to whatever size you want to resize to
	width = (int)(windowsize.right * scaleFactor);

	src.create(height, width, CV_8UC4);


	src.create(height, width, CV_8UC4);

	// create a bitmap
	hbwindow = CreateCompatibleBitmap(hwindowDC, width, height);
	bi.biSize = sizeof(BITMAPINFOHEADER);    //http://msdn.microsoft.com/en-us/library/windows/window/dd183402%28v=vs.85%29.aspx
	bi.biWidth = width;
	bi.biHeight = -height;  //this is the line that makes it draw upside down or not
	bi.biPlanes = 1;
	bi.biBitCount = 32;
	bi.biCompression = BI_RGB;
	bi.biSizeImage = 0;
	bi.biXPelsPerMeter = 0;
	bi.biYPelsPerMeter = 0;
	bi.biClrUsed = 0;
	bi.biClrImportant = 0;

	// use the previously created device context with the bitmap
	SelectObject(hwindowCompatibleDC, hbwindow);
	// copy from the window device context to the bitmap device context
	StretchBlt(hwindowCompatibleDC, 0, 0, width, height, hwindowDC, 0, 0, srcwidth, srcheight, SRCCOPY); //change SRCCOPY to NOTSRCCOPY for wacky colors !
	GetDIBits(hwindowCompatibleDC, hbwindow, 0, height, src.data, (BITMAPINFO *)&bi, DIB_RGB_COLORS);  //copy from hwindowCompatibleDC to hbwindow

	// avoid memory leak
	DeleteObject(hbwindow); DeleteDC(hwindowCompatibleDC); ReleaseDC(hwnd, hwindowDC);

	return src;

}

long captureVids(InputProcessing ip, string folderName, float scaleFactor) {
	string windowName = "Capture";
	namedWindow(windowName);
	Mat rec, pause;
	try {
		rec = imread("img/recording.png");
		pause = imread("img/paused.png");

	}
	catch (int n) {
		printf("Error: file \"img/recording.png\" or \"img/pause.png\" not found!\n");
		return 0;
	}
	imshow(windowName, rec);
	HWND hDesktopWnd;
	HDC hDesktopDC;
	hDesktopWnd = GetDesktopWindow();
	hDesktopDC = GetDC(hDesktopWnd);
	Size screen = getScreenRes(scaleFactor);
	HBITMAP hbDesktop = CreateCompatibleBitmap(hDesktopDC, screen.width, screen.height);
	Mat screenFrame, screenFrame3C;
	Mat cameraFrame;
	// go until user stops or integer overflows
	long i = 1;
	for (; i > 0; i++){
		screenFrame = hwnd2mat(hDesktopWnd, scaleFactor);
		cvtColor(screenFrame, screenFrame3C, CV_BGRA2BGR);
		cameraFrame = ip.getNextFrame(0);
		string fileNum = ip.formatedIntToStr(i, 5);
  		string filenam = folderName + "/scr_" + fileNum + ".png";
		bool b1 = imwrite(folderName + "/scr_" + fileNum + ".png", screenFrame);
		bool b2 = imwrite(folderName + "/cam_" + fileNum + ".png", cameraFrame);
		char c = waitKey(20);
		//escape and enter
		if (c == 27 || c == 13) {
			break;
		
		}
		//spacebar
		else if (c == 32) {
			imshow(windowName, pause);
			while (true) {
				c = waitKey(20);
				if (c == 27 || c == 13) {
					destroyAllWindows();
					return i;
				}
				if (c == 32) {
					imshow(windowName, rec);
					break;
					
				}
			}
		}
	}
	destroyAllWindows();
	return i;
}

char * strToCharArr(string s) {
	char *a = new char[s.size() + 1];
	a[s.size()] = 0;
	memcpy(a, s.c_str(), s.size());
	return a;
}

void processTrackingData(InputProcessing ip, string folderName, long lastFrameNum, const char * trackFile) {
	ofstream f;
	f.open(trackFile);
	for (long i = 1; i <= lastFrameNum; i++) {
		string fileNum = ip.formatedIntToStr(i, 5);
		string fileName = folderName + "/cam_" + fileNum + ".png";
		char * fileNamePrehistoric = strToCharArr(fileName);
		Mat frame = imread(fileNamePrehistoric);
		Mat gray;
		cvtColor(frame, gray, CV_BGR2GRAY);
		vector<Point> features = ip.getFeatures(gray);
		if (features.size() == 0) {
			f << "0 0 0 0 0 0 0 0 0" << endl;
			std::remove(fileNamePrehistoric);
			continue;
		}
		while (features.size() < 6)
			features.push_back(features[0]);
			
		vector<double> derf = ip.getDerivativeFeatures(features, gray.size());

		f << derf[0] << ' '
			<< derf[1] << ' '
			<< derf[2] << ' '
			<< derf[3] << ' '
			<< derf[4] << ' '
			<< derf[5] << ' '
			<< derf[6] << ' '
			<< derf[7] << ' '
			<< derf[8] << endl;
		std::remove(fileNamePrehistoric);
		printf("frame %d / %d \n", i, lastFrameNum);
	}
	f.close();
	
}

bool processOutput(InputProcessing ip, long trainCount, const char * trainFile, long trackCount, const char * trackFile, 
	const char * outputFile, string outputImagesFolderName, Size screenSize) {
	int num_input = 9;
	int num_hidden = 6;
	int num_output = 2;

	Mat layerSizes(3, 1, CV_16S);
	short * ptr = layerSizes.ptr<short>();
	ptr[0] = num_input;
	ptr[1] = num_hidden;
	ptr[2] = num_output;


	Ptr<ml::ANN_MLP> mlp = ml::ANN_MLP::create();

	mlp->setLayerSizes(layerSizes);
	mlp->setActivationFunction(cv::ml::ANN_MLP::SIGMOID_SYM);
	//mlp->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER | TermCriteria::EPS, 10000, .02));
	Mat_<float> train(trainCount, 11, CV_32F);
	int t = ip.loadMatFromCSVFile(trainFile, train, 11, trainCount);
	if (t < 0) {
		return false;
	}
	Mat inputs = train.colRange(0, 9);
	Mat targets(trainCount, 2, CV_32F);

	//not sure why this is necessary, but targets = train.colRange(12,14) is not accepted by ANN_MLP::train()
	Mat targets2 = train.colRange(9, 11);
	MatIterator_<float> it = targets.begin<float>();
	MatIterator_<float> it2 = targets2.begin<float>();
	for (; it != targets.end<float>(); it++, it2++) {
		(*it) = (*it2);
	}

	if (!mlp->train(inputs, ml::ROW_SAMPLE, targets)) {
		return false;
	}
	Mat track(trackCount, 9, CV_32F);
	t = ip.loadMatFromCSVFile(trackFile, track, 9, trackCount);
	Mat samples = track.colRange(0, 9);
	Mat out;
	mlp->predict(samples, out);
	ofstream f;
	f.open(outputFile);
	it = out.begin<float>();
	long i = 1;

	//save output to both outputFile and draw guessed point to images in outputImagesFolderName folder
	while (it != out.end<float>()) {
		f << (*it) << " ";
		double x = (*it);
		it++;
		f << (*it) << endl;
		double y = (*it);
		it++;

		//normalize x and y
		if (x > 1) {
			x = 1;
		}
		else if (x < 0) {
			x = 0;
		}
		if (y > 1) {
			y = 1;
		}
		else if (y < 0) {
			y = 0;
		}
		x *= screenSize.width;
		y *= screenSize.height;
		string fileNum = ip.formatedIntToStr(i, 5);

		string fileName = outputImagesFolderName + "/scr_" + fileNum + ".png";
		Mat m = imread(fileName);
		circle(m, Point(x, y), 10, Scalar(255, 25, 255),3);
		imwrite(fileName, m);
		i++;
	}
	f.close();
	return 0;

}
