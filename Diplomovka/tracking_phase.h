#include <windows.h>

#include "inputprocessing.h"

Size getScreenRes();
Mat  hwnd2mat(HWND hwnd);
long captureVids(InputProcessing ip, String fileName);