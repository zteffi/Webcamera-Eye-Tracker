cmake_minimum_required(VERSION 3.5)
project(Diplomovka)
find_package( OpenCV REQUIRED )
set (OpenCV_SHARED OFF)
set(SOURCE_FILES
	eyetracker.cpp
	inputprocessing.h
	inputprocessing.cpp
	testing.h
	testing.cpp
	tracking_phase.h
	tracking_phase.cpp
	training_phase.h
	training_phase.cpp
	)
	
add_executable(Diplomovka
	${SOURCE_FILES}
	)
	
target_link_libraries(Diplomovka ${OpenCV_LIBS})

file(COPY ${CMAKE_SOURCE_DIR}/cascades DESTINATION ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE})


