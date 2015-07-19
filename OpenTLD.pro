#-------------------------------------------------
#
# Project created by QtCreator 2015-06-08T10:28:52
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = OpenTLD
TEMPLATE = app


SOURCES +=\
        mainwindow.cpp \
    src/FerNNClassifier.cpp \
    src/LKTracker.cpp \
    src/run_tld.cpp \
    src/TLD.cpp \
    src/tld_utils.cpp

HEADERS  += mainwindow.h \
    include/FerNNClassifier.h \
    include/LKTracker.h \
    include/TLD.h \
    include/tld_utils.h

FORMS    += mainwindow.ui
win32{
    INCLUDEPATH+="D:\Program Files\opencv\mingw\install\include\opencv"   \
                    "D:\Program Files\opencv\mingw\install\include\opencv2"   \
                    "D:\Program Files\opencv\mingw\install\include"
    LIBS+="D:\Program Files\opencv\mingw\install\x64\mingw\lib\libopencv_calib3d2411.dll.a" \
        "D:\Program Files\opencv\mingw\install\x64\mingw\lib\libopencv_contrib2411.dll.a" \
        "D:\Program Files\opencv\mingw\install\x64\mingw\lib\libopencv_core2411.dll.a" \
        "D:\Program Files\opencv\mingw\install\x64\mingw\lib\libopencv_features2d2411.dll.a" \
        "D:\Program Files\opencv\mingw\install\x64\mingw\lib\libopencv_flann2411.dll.a" \
        "D:\Program Files\opencv\mingw\install\x64\mingw\lib\libopencv_gpu2411.dll.a" \
        "D:\Program Files\opencv\mingw\install\x64\mingw\lib\libopencv_highgui2411.dll.a" \
        "D:\Program Files\opencv\mingw\install\x64\mingw\lib\libopencv_imgproc2411.dll.a" \
        "D:\Program Files\opencv\mingw\install\x64\mingw\lib\libopencv_legacy2411.dll.a" \
        "D:\Program Files\opencv\mingw\install\x64\mingw\lib\libopencv_ml2411.dll.a" \
        "D:\Program Files\opencv\mingw\install\x64\mingw\lib\libopencv_nonfree2411.dll.a" \
        "D:\Program Files\opencv\mingw\install\x64\mingw\lib\libopencv_objdetect2411.dll.a" \
        "D:\Program Files\opencv\mingw\install\x64\mingw\lib\libopencv_ocl2411.dll.a" \
        "D:\Program Files\opencv\mingw\install\x64\mingw\lib\libopencv_photo2411.dll.a" \
        "D:\Program Files\opencv\mingw\install\x64\mingw\lib\libopencv_stitching2411.dll.a" \
        "D:\Program Files\opencv\mingw\install\x64\mingw\lib\libopencv_superres2411.dll.a" \
        "D:\Program Files\opencv\mingw\install\x64\mingw\lib\libopencv_ts2411.a" \
        "D:\Program Files\opencv\mingw\install\x64\mingw\lib\libopencv_video2411.dll.a" \
        "D:\Program Files\opencv\mingw\install\x64\mingw\lib\libopencv_videostab2411.dll.a"
}
