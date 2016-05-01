
#include "inputprocessing.h"

InputProcessing::InputProcessing(int inputType, bool DEBUG_MODE) :
	faceCascade(CascadeClassifier()),
	inputType(inputType),
	DEBUG_MODE(DEBUG_MODE),
	eyeCascadeGlasses(CascadeClassifier()),
	eyeCascade(CascadeClassifier())

{
	
	printf("loading \n");
	if (!faceCascade.load("cascades\\haarcascade_frontalface_alt.xml")) {
		printf("--(!)File not found faceCascade\n"); exit(-11);
	}
	if (!eyeCascadeGlasses.load("cascades\\haarcascade_eye_tree_eyeglasses.xml")) {
		printf("--(!)File not found eyeCascadeGlasses\n"); exit(-12);
	}
	if (!eyeCascade.load("cascades\\haarcascade_eye.xml")) {
		printf("--(!)File not found eyeCascade\n"); exit(-13);
	}
	if (inputType < 1 || inputType > 4) {
		printf("--(!)Input type %i not specified\n", inputType); exit(-15);
	} 
	else if (inputType == INPUT_TYPE_CAMERA_INPUT) {
		// set default camera
		cap = VideoCapture(0);
		if (!cap.isOpened()) {
			printf("--(!)Camera 0 not available\n");
		}
	} 
	else if (inputType == INPUT_TYPE_GI4E_DB) {
		/*
		load ground truth

		Format for GI4E image_labels.txt:
		xxx_yy.png	x1 y1 x2 y2 x3 y3 x4 y4 x5 y5 x6 y6
		The first point (x1,y1) is the external corner of the left user's eye. The second point is the centre of the left iris.
		The third one is the internal corner of the left eye. The other three points are internal corner, iris centre and
		external corner of the right eye.

		*/
		
		ifstream file("../GI4E/labels/image_labels.txt");
		string line;
		if (file.is_open()) {
			while (getline(file, line)) {
				try {
					istringstream iss(line);
					string filename, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6;
					getline(iss, filename, '\t');
					getline(iss, x1, '\t');
					getline(iss, y1, '\t');
					getline(iss, x2, '\t');
					getline(iss, y2, '\t');
					getline(iss, x3, '\t');
					getline(iss, y3, '\t');
					getline(iss, x4, '\t');
					getline(iss, y4, '\t');
					getline(iss, x5, '\t');
					getline(iss, y5, '\t');
					getline(iss, x6, '\t');
					getline(iss, y6, '\t');
					vector<Point2f> v;
					
					v.push_back(Point2f(stof(x1), stof(y1)));
					v.push_back(Point2f(stof(x2), stof(y2)));
					v.push_back(Point2f(stof(x3), stof(y3)));
					v.push_back(Point2f(stof(x4), stof(y4)));
					v.push_back(Point2f(stof(x5), stof(y5)));
					v.push_back(Point2f(stof(x6), stof(y6)));
					v.shrink_to_fit();
					labels.push_back(v);
				}
				catch (Exception e) {
					printf("--(!)Error while parsing /GI4E/labels/image_labels.txt\n");
				}
			}
			labels.shrink_to_fit();
			file.close();
		}
	}

}
/* formats n as string of length len with leading zeroes  */
string formatedStrToInt(unsigned int n, unsigned int len) {
	string res(len, '0');
	for (string::reverse_iterator i = res.rbegin(); i != res.rend(); i++) {
		*i = n % 10 + '0';
		n /= 10;
	}
	return res;
}

Mat InputProcessing::getNextFrame(unsigned long frameNum) {
	Mat frame;
	switch (inputType) {
	case INPUT_TYPE_VIDEO_INPUT:
		if (cap.get(CV_CAP_PROP_FRAME_COUNT) <= frameNum) {
			break;
		}
	case INPUT_TYPE_CAMERA_INPUT:
		cap >> frame; //get next frame from camera/video
		break;
	case INPUT_TYPE_BIOID_DB:
		// when we used all photos in db, return empty matrix
		if (frameNum < 340) {
			frame = imread("../BioID-FaceDatabase-V1.2/BioID_" + to_string(frameNum + 1181) + ".pgm");
		}
		break;
	case INPUT_TYPE_GI4E_DB:
	{
		unsigned int personCount = frameNum / 12 + 1;
		unsigned int frameCount = frameNum % 12 + 1;
		// when we used all photos in db, return empty matrix
		if (personCount <= 103) {
			string filename;
			filename = formatedStrToInt(personCount, 3) + "_" +
				formatedStrToInt(frameCount, 2) + ".png";
			frame = imread("../GI4E/images/" + filename);
		}
		break;
	}
	default:
		printf("--(!)Input type %i not specified\n", inputType); exit(-15);
	} 
	return frame;
}



Mat InputProcessing::getRedChannelMatrix(const Mat  frame) {
	if (frame.channels() == 1) {
		return frame.clone();
	}
	else if (frame.channels() == 3) {
		Mat channel[3];
		split(frame, channel);
		/* if input is in rgb, we use red channel, since it maximizes contrast
		*  between iris and sclera, regardless of person's eye color.
		*/
		return channel[2];
	}
	printf("--(!)Unsupported number of channels : " + frame.channels());
	exit(101);
}


Rect InputProcessing::getFacePosition(const Mat  frame) {
	vector<Rect> faces(1);
	int slidingWindowSize = min(frame.rows,frame.cols)/12;
	faceCascade.detectMultiScale(frame, faces, 1.1, 3, CV_HAAR_FIND_BIGGEST_OBJECT, Size(slidingWindowSize, slidingWindowSize));

	if (faces.size() >= 1) {
		return  faces[0];
	}
	return Rect(0, 0, 0, 0);
}

Rect InputProcessing::getLeftEyePosition(Mat frame, Rect facePosition) {
	Size minS((int)(facePosition.width*.15), (int)(facePosition.height*.15));
	Size maxS((int)(facePosition.width*.4), (int)(facePosition.height*.4));
	vector<Rect> eyes(1);
	facePosition.width /= 2;
	facePosition.height /= 2;
	facePosition.x += facePosition.width;
	facePosition.y += facePosition.height / 3;
	Mat face(frame, facePosition);
	eyeCascade.detectMultiScale(face, eyes, 1.05, 3, CV_HAAR_FIND_BIGGEST_OBJECT, minS, maxS);
	if (eyes.size() == 0) {
		return Rect(0, 0, 0, 0);
	}
	eyes.at(0).x += facePosition.x;
	eyes.at(0).y += facePosition.y;
	//from empirical observations, cascades tend to crop right parts for both eyes, so we add width
	eyes.at(0).width += (int) ceil(eyes.at(0).width *.125);
	return eyes.at(0);
}

Rect InputProcessing::getRightEyePosition(Mat frame, Rect facePosition) {
	Size minS((int)(facePosition.width*.15), (int)(facePosition.height*.15));
	Size maxS((int)(facePosition.width*.4), (int)(facePosition.height*.4));
	vector<Rect> eyes(1);
	facePosition.width /= 2;
	facePosition.height /= 2;
	facePosition.y += facePosition.height / 3;
	Mat  face(frame, facePosition);
	eyeCascade.detectMultiScale(face, eyes, 1.05, 3, CV_HAAR_FIND_BIGGEST_OBJECT,minS,maxS);
	if (eyes.size() == 0) {
		return  Rect(0,0,0,0);
	}
	eyes.at(0).x += facePosition.x;
	eyes.at(0).y += facePosition.y;
	//from empirical observations, cascades tend to crop right parts for both eyes, so we add width
	eyes.at(0).width += (int) ceil(eyes.at(0).width *.125);
	return eyes.at(0);
}



Point2i InputProcessing::getLeftEyeCorner(Mat gray, Rect leftEye) {
	Rect leftEyeCorner = leftEye;
	//omit top 1/4 of image
	leftEyeCorner.y += (int) (leftEyeCorner.height * .25);
	leftEyeCorner.height /= 2;

	leftEyeCorner.width *= .5;

	Mat im = Mat(gray, leftEyeCorner);
	vector<Point2i> features;
	//	GaussianBlur(im, im, Size(3, 3), 0, 0);
	goodFeaturesToTrack(im, features, 15, .15, leftEyeCorner.height / 8);

	double minDist = DBL_MAX, minIndex = -1, i = 0;
	
	for (Point2i p : features) {
		//ydist = distnace from middle of inner edge of leftEye rectangle
		double ydist = (p.y - (leftEye.height / 4));
		double dist = p.x * p.x + ydist * ydist * 4;
		// y difference is less likely for eye corner
		if (dist < minDist) {
			minDist = dist;
			minIndex = i;
		}
		i++;
	}
	
	if ( minIndex >= 0 ) {
		Point2i res = features[minIndex] + Point2i(leftEyeCorner.x, leftEyeCorner.y);
		if (DEBUG_MODE) {
			//circle(drawFrame, Point2i(leftEyeCorner.x, leftEyeCorner.y + leftEye.height / 4), 1, Scalar(10, 10, 255));
		}
		return res;
	}
	
	return Point2i(-1,-1);
}

Point2i InputProcessing::getRightEyeCorner(Mat gray, Rect rightEye) {
	Rect rightEyeCorner = rightEye;
	//omit top 1/4 of image
	rightEyeCorner.y += rightEyeCorner.height * .25;
	rightEyeCorner.height /= 2;
	rightEyeCorner.x += .5 * rightEyeCorner.width;
	rightEyeCorner.width *= .5;

	Mat im = Mat(gray, rightEyeCorner);
	vector<Point2i> features;
	//	GaussianBlur(im, im, Size(3, 3), 0, 0);
	goodFeaturesToTrack(im, features, 15, .15, rightEyeCorner.height / 16);
	

	double minDist = DBL_MAX, minIndex = -1, i = 0;

	for (Point2i p : features) {
		//ydist = distnace from middle of inner edge of leftEye rectangle
		double ydist = (p.y - (rightEye.height / 4));
		double xdist = (p.x - (rightEye.width / 2));
		double dist = xdist * xdist + ydist * ydist * 4;

		if (dist < minDist) {
			minDist = dist;
			minIndex = i;
		}
		i++;
	}

	if (minIndex >= 0) {
		Point2i res = features[minIndex] + Point2i(rightEyeCorner.x, rightEyeCorner.y);
		if (DEBUG_MODE) {
			//circle(drawFrame, Point2i(rightEye.x + rightEye.width, rightEye.y + rightEye.height / 2), 1, Scalar(255, 10, 10));
		}
		return res;
	}

	return Point2i(-1, -1);
}



Point InputProcessing::getGroundTruth(int frameCount, int flag) {
	if (inputType != INPUT_TYPE_GI4E_DB) {
		return Point(0, 0);
	}
	return labels.at(frameCount).at(flag);
}

Point2d InputProcessing::getEyeCenter(Mat frame, Rect eye) {

	//find derivatives
	Mat ROI = frame(eye);
	Mat gx, gy;
	GaussianBlur(ROI, ROI, Size(3, 3), 0, 0);
	Scharr(ROI, gx, CV_32F, 1, 0);
	Scharr(ROI, gy, CV_32F, 0, 1);

	//initialize variables
	Point cmaxPoint;
	double cmax = DBL_MIN;
	double c = 0;
	Mat cmap(gx.size(), CV_64FC1, Scalar(0));

	//compute cmap field
	float * px, *py;
	unsigned char * w;
	double * pc;

	//for each vector g = gradient with origin at ROI(j,i)
	for (int i = 0; i < gx.rows; i++) {
		px = gx.ptr<float>(i);
		py = gy.ptr<float>(i);

		for (int j = 0; j < gx.cols; j++) {
			float plen = sqrtf(px[j] * px[j] + py[j] * py[j]);
			if (plen == 0)
				continue;
			px[j] /= plen;
			py[j] /= plen;

			double dxt, dyt = 0; //d^T.x, d^T.y where d is vector from c to origin of g
			if (plen < 300) {
				continue;
			}

			//for each possibl centre c = ROI(l,k)
			for (int k = 0; k < gx.rows; k++) {
				dxt = -k + i;
				pc = cmap.ptr<double>(k);
				w = ROI.ptr<unsigned char>(k);
				for (int l = 0; l < gx.cols; l++) {
					if (i == k && j == l)
						continue;
					dyt = l - j;
					double dlen = sqrt(dxt*dxt + dyt*dyt);
					double cz = px[j] * dyt / dlen - py[j] * dxt / dlen; //cross product
					pc[l] += cz*cz * (255 - w[l]); // inverted intensity value, since pupil is usually darker

				}
			}


		}

	}
	for (int i = 8; i < gx.rows - 8; i++) {
		pc = cmap.ptr<double>(i);

		for (int j = 8; j < gx.cols - 8; j++) {
			if (pc[j] > cmax) {
				cmaxPoint.x = j;
				cmaxPoint.y = i;
				cmax = pc[j];
			}
		}
	}
	cmaxPoint += Point(eye.x, eye.y);
	return cmaxPoint;
}





/*
  finds minimal intensity for center 1/9 of ROI after shrinking image width and height by shrinkFactor
  */
Point InputProcessing::getPupilPointFromIntensity(Mat ROI,  double shrinkFactor) {
	if (ROI.rows / shrinkFactor < 3 || ROI.cols / shrinkFactor < 3) {
		return Point(0, 0);
	}
	Size shrinkSize = ROI.size();
	shrinkSize.width = floor(shrinkSize.width / shrinkFactor);
	shrinkSize.height = floor(shrinkSize.height / shrinkFactor);
	Mat shrink;
	resize(ROI, shrink, shrinkSize);
	Point c(0.0, 0.0);
	unsigned char * v, *vp, * vm;
	unsigned char minIntensity = 255;

	//find minimal intensity of shrinked image
	int i = shrink.rows / 3;
	vm = shrink.ptr<unsigned char>(i - 1);
	v = shrink.ptr<unsigned char>(i);
	vp = shrink.ptr<unsigned char>(i + 1);
	for (; i < shrink.rows - shrink.rows / 3; i++) {
	
		for (int j = shrink.cols / 3; j < shrink.cols - shrink.cols / 3; j++) {
			int val = v[j] + v[j + 1] + v[j - 1];
			if (val < minIntensity) {
				minIntensity = val;
				c.x = (int) (j * shrinkFactor);
				c.y = (int) (i * shrinkFactor);
			}
		}
		vm = v;
		v = vp;
		vp = shrink.ptr<unsigned char>(i + 1);
	}
	return c;

}


Point getPupilPointFromGradient(Mat gx) {
	double maxVal = FLT_MIN;
	Point2i maxP(0,0);
	double minVal = FLT_MAX;
	Point2i minP(0, 0);
	double * v;
	for (int i = 0; i < gx.rows; i++) {
		v = gx.ptr<double>(i);
		for (int j = 0; j < gx.cols; j++) {
			if (v[j] > maxVal) {
				maxVal = v[j];
				maxP.x = j;
				maxP.y = i;
			}
			else if (v[j] < minVal) {
				minVal = v[j];
				minP.x = j;
				minP.y = i;
			}
		}
		
	}
	return Point((minP.x + maxP.x) / 2, (minP.y + maxP.y) / 2);
}

/*for center c and gradients in gx and gy, give sum of square errors of distance*/
double getFunctionResponseForCenter(Mat gx, Mat gy, Point c) {
	double res = 0;
	int ROIsize = (gx.rows * gx.cols);
	double * px, *py;
	try{
		for (int i = 0; i < gx.rows; i++) {
			px = gx.ptr<double>(i);
			py = gy.ptr<double>(i);
			for (int j = 0; j < gx.cols; j++) {
				double plen = sqrtf(px[j] * px[j] + py[j] * py[j]);


				if (plen < 300) {
					continue;
				}
				//d^T.x, d^T.y where d is vector from c to origin of g
				double dxt = i - c.x;
				double dyt = j - c.y;

				double dlen = sqrt(dxt*dxt + dyt*dyt);
				if (dlen == 0) {
					continue;
				}
				double cz = px[j] * dyt / dlen - py[j] * dxt / dlen; //cross product
				cz /= ROIsize;
				res += cz*cz;

			}

		}
	}
	catch (int e) {
		cout << e << '\n';

	}
	return res;
}


bool InputProcessing::saveFeatures(ofstream & file, int x, int y, Size screenSize) {
	
	Mat frame = getNextFrame(frameCount++);
	Mat gray;
	
	cvtColor(frame, gray, CV_BGR2GRAY);

	//locate face
	
	vector<Point> features = getFeatures(gray);
	if (features.size() != 6)
		return false;

	if (DEBUG_MODE) {
		circle(frame, features[0], 2, Scalar(10, 255, 255), -1, 8, 0);	
		circle(frame, features[1], 2, Scalar(20, 210, 21), -1, 8, 0);
		circle(frame, features[2], 2, Scalar(10, 255, 255), -1, 8, 0);
		circle(frame, features[3], 2, Scalar(10, 255, 255), -1, 8, 0);
		circle(frame, features[4], 2, Scalar(20, 210, 21), -1, 8, 0);
		circle(frame, features[5], 2, Scalar(10, 255, 255), -1, 8, 0);
		
		imshow(WINDOW_NAME, frame);
	}
	
	file << features[0].x / (double)gray.cols << ' ' << features[0].y / (double)gray.rows << ' '
		<< features[1].x / (double)gray.cols << ' ' << features[1].y / (double)gray.rows << ' '
		<< features[2].x / (double)gray.cols << ' ' << features[2].y / (double)gray.rows << ' '
		<< features[3].x / (double)gray.cols << ' ' << features[3].y / (double)gray.rows << ' '
		<< features[4].x / (double)gray.cols << ' ' << features[4].y / (double)gray.rows << ' '
		<< features[5].x / (double)gray.cols << ' ' << features[5].y / (double)gray.rows << ' '
		<< x / (double) screenSize.width << ' '
		<< y / (double) screenSize.height
		<< endl;


	return true;
}

vector<Point> InputProcessing::getFeatures(Mat gray) {
	vector<Point> features;
	Rect face = getFacePosition(gray);
	
	if (face.width == 0) {
		return features;
	}
	Rect leftEye = getLeftEyePosition(gray, face);
	Rect rightEye = getRightEyePosition(gray, face);

	if (leftEye.width == 0 || rightEye.width == 0) {
		return features;
	}


	features.push_back(getLeftEyeCorner(gray, rightEye));
	features.push_back(getEyeCenter(gray, rightEye));
	features.push_back(getRightEyeCorner(gray, rightEye));
	features.push_back(getLeftEyeCorner(gray, leftEye));
	features.push_back(getEyeCenter(gray, leftEye));
	features.push_back(getRightEyeCorner(gray, leftEye));

	if (features.size() < 6) {
		features.clear();
		return features;
	}

	if (features[0].x == -1 || features[1].x == -1
		|| features[2].x == -1 || features[3].x == -1 
		|| features[4].x == -1 || features[5].x == -1) {
		features.clear();
	}

	return features;
}

int InputProcessing:: processTrainingFile(const char * inputFile, int inputCount, const char  * trackFile, int trackCount, const char  * outputFile) {
	
	int num_input = 12;
	int num_hidden = 30;
	int nun_hidden_2 = 12;
	int num_output = 2;

	
	const double desired_error = (const double) .05;
	const unsigned int max_epochs = 50000;
	const unsigned int epochs_between_reports = 1000;

	Mat layerSizes(3,1,CV_16S);
	short * ptr = layerSizes.ptr<short>();
	ptr[0] = num_input;
    ptr[1] = num_hidden;
	ptr[2] = num_output;


	Ptr<ml::ANN_MLP> mlp = ml::ANN_MLP::create();
	
	mlp->setLayerSizes(layerSizes);
	mlp->setActivationFunction(cv::ml::ANN_MLP::SIGMOID_SYM);
	//mlp->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER | TermCriteria::EPS, 10000, .02));

	Mat_<float> train(inputCount, 14, CV_32F);
	int t = loadMatFromCSVFile(inputFile, train, 14, inputCount);
	if (t < 0) {
		return t;
	}
	Mat inputs = train.colRange(0, 12);
	Mat targets(inputCount, 2, CV_32F); 

	//not sure why this is necessary, but targets = train.colRange(12,14) is not accepted by ANN_MLP::train()
	Mat targets2 = train.colRange(12, 14);
	MatIterator_<float> it = targets.begin<float>();
	MatIterator_<float> it2 = targets2.begin<float>();
	for (; it != targets.end<float>(); it++, it2++) {
		(*it) = (*it2);
	}
	if (!mlp->train(inputs, ml::ROW_SAMPLE, targets)) {
		return -2;
	}
	//!TODO
	Mat track(trackCount, 14, CV_32F);
	t = loadMatFromCSVFile(trackFile, track, 14, trackCount);
	Mat samples = track.colRange(0, 12);
	Mat out;
	mlp->predict(samples, out);
	ofstream f;
	f.open(outputFile);
	it = out.begin<float>();
	while (it != out.end<float>()) {
		f << (*it) << " ";
		it++;
		f << (*it) << endl;
		it++;
	}
	f.close();
	return 0;

	
}

int InputProcessing::loadMatFromCSVFile(const char* filename, Mat & mat, int numAttr, int numLines) {
	FILE * f = fopen(filename, "r");
	if (!f) {
		printf("ERROR: cannot read file %s\n", filename);
		return -1; // all not OK
	}

	MatIterator_<float> it, end;
	
	for (it = mat.begin<float>(), end = mat.end<float>(); it != end; ++it) {
		float d;
		fscanf(f, "%f,", &d);
		(*it) = d;
	}
	return 0;
}

/* if input is live video stream, specifie camera number */
void InputProcessing::setCamera(int deviceNum) {

}

/* if input is a video file, set file */
void InputProcessing::setVideo(string file) {

}

Point InputProcessing::getApproximateEyeCenter(Mat frame, Rect eye) {
	return Point(-1, -1);
}


