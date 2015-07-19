#include <opencv2/opencv.hpp>
#include "include/tld_utils.h"
#include <iostream>
#include <sstream>
#include "include/TLD.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <QDebug>
using namespace cv;
using namespace std;
//Global variables
Rect box;
bool drawing_box = false;
bool gotBB = false;
bool tl = true;
bool rep = false;
bool fromfile=false;
string video;

void readBB(char* file){
    ifstream bb_file (file);
    string line;
    getline(bb_file,line);
    istringstream linestream(line);
    string x1,y1,x2,y2;
    getline (linestream,x1, ',');
    getline (linestream,y1, ',');
    getline (linestream,x2, ',');
    getline (linestream,y2, ',');
    int x = atoi(x1.c_str());// = (int)file["bb_x"];
    int y = atoi(y1.c_str());// = (int)file["bb_y"];
    int w = atoi(x2.c_str())-x;// = (int)file["bb_w"];
    int h = atoi(y2.c_str())-y;// = (int)file["bb_h"];
    box = Rect(x,y,w,h);
}
//bounding box mouse callback
void mouseHandler(int event, int x, int y, int flags, void *param){
    switch( event ){
    case CV_EVENT_MOUSEMOVE:
        if (drawing_box){
            box.width = x-box.x;
            box.height = y-box.y;
        }
        break;
    case CV_EVENT_LBUTTONDOWN:
        drawing_box = true;
        box = Rect( x, y, 0, 0 );
        break;
    case CV_EVENT_LBUTTONUP:
        drawing_box = false;
        if( box.width < 0 ){
            box.x += box.width;
            box.width *= -1;
        }
        if( box.height < 0 ){
            box.y += box.height;
            box.height *= -1;
        }
        gotBB = true;
        break;
    }
}

void print_help(char** argv){
    printf("use:\n     %s -p /path/parameters.yml\n",argv[0]);
    printf("-s    source video\n-b        bounding box file\n-tl  track and learn\n-r     repeat\n");
}

void read_options(int argc, char** argv,VideoCapture& capture,FileStorage &fs){
    for (int i=0;i<argc;i++){
        if (strcmp(argv[i],"-b")==0){
            if (argc>i){
                readBB(argv[i+1]);
                gotBB = true;
            }
            else
                print_help(argv);
        }
        if (strcmp(argv[i],"-s")==0){
            if (argc>i){
                video = string(argv[i+1]);
                capture.open(video);
                fromfile = true;
            }
            else
                print_help(argv);

        }
        if (strcmp(argv[i],"-p")==0){
            if (argc>i){
                fs.open(argv[i+1], FileStorage::READ);
            }
            else
                print_help(argv);
        }
        if (strcmp(argv[i],"-no_tl")==0){
            tl = false;
        }
        if (strcmp(argv[i],"-r")==0){
            rep = true;
        }
    }
}

int main(int argc, char * argv[]){
    VideoCapture capture;
    capture.open(0);
    FileStorage fs;
    //Read options
    //  read_options(argc,argv,capture,fs);
    fs.open("D:\\opencv_project\\OpenTLD\\parameters.yml", FileStorage::READ);
    if(fs.isOpened())
    {
        qDebug() << "file is opened";
    }
    //Init camera
    if (!capture.isOpened())
    {
        cout << "capture device failed to open!" << endl;
        return 1;
    }

    //TLD framework
    TLD tld;
    //Read parameters file
    tld.read(fs.getFirstTopLevelNode());
    Mat frame;
    Mat last_gray;
    Mat first;
    if (fromfile){
        capture >> frame;
        cvtColor(frame, last_gray, CV_RGB2GRAY);
        frame.copyTo(first);
    }else{
        capture.set(CV_CAP_PROP_FRAME_WIDTH,340);
        capture.set(CV_CAP_PROP_FRAME_HEIGHT,240);
    }
#ifndef AUTO
    //Register mouse callback to draw the bounding box
    cvNamedWindow("TLD",CV_WINDOW_AUTOSIZE);
    cvSetMouseCallback( "TLD", mouseHandler, NULL );
    ///Initialization
GETBOUNDINGBOX:
    while(!gotBB)
    {
        if (!fromfile){
            capture >> frame;
        }
        else
            first.copyTo(frame);
        cvtColor(frame, last_gray, CV_RGB2GRAY);

        imwrite("D:\\opencv_project\\OpenTLD\\first\\first.jpg",frame);

        drawBox(frame,box);
        imshow("TLD", frame);
        if (cvWaitKey(33) == 'q')
            return 0;


    }

    if (min(box.width,box.height)<(int)fs.getFirstTopLevelNode()["min_win"]){
        cout << "Bounding box too small, try again." << endl;
        gotBB = false;
        goto GETBOUNDINGBOX;
    }
    //Remove callback
    cvSetMouseCallback( "TLD", NULL, NULL );


    FileStorage box_fs("D:\\opencv_project\\OpenTLD\\first\\box.yml", FileStorage::WRITE);
    box_fs << "box" << box ;
    box_fs.release();
#endif
#ifdef AUTO
    capture >> frame;
    cvtColor(frame, last_gray, CV_RGB2GRAY);
    first = imread("D:\\opencv_project\\OpenTLD\\first\\first.jpg");
    cvtColor(first, first, CV_RGB2GRAY);
    FileStorage box_fs("D:\\opencv_project\\OpenTLD\\first\\box.yml", FileStorage::READ);
    box_fs["box"] >>  box;
    box_fs.release();
#endif
    printf("Initial Bounding Box = x:%d y:%d h:%d w:%d\n",box.x,box.y,box.width,box.height);
    //Output file
    FILE  *bb_file = fopen("bounding_boxes.txt","w");
    //TLD initialization
#ifndef AUTO

    tld.init(last_gray,box,bb_file);
#else
    tld.init(first,box,bb_file);
#endif

    ///Run-time
    Mat current_gray;
    BoundingBox pbox;
    vector<Point2f> pts1;
    vector<Point2f> pts2;
#ifndef AUTO
    bool status=true;
#else
    bool status = false;
#endif
    int frames = 1;
    int detections = 1;
    char strFrame[64] = "hello";
    double toc;
REPEAT:
    while(capture.read(frame)){

        double t = (double)cvGetTickCount();
        //get frame
        cvtColor(frame, current_gray, CV_RGB2GRAY);
        //Process Frame
        tld.processFrame(last_gray,current_gray,pts1,pts2,pbox,status,tl,bb_file);
        //Draw Points
        if (status){
            drawPoints(frame,pts1);//聚合框内的网点
            drawPoints(frame,pts2,Scalar(0,255,0));//检测到网点
            drawBox(frame,pbox);
            detections++;
        }
        //Display
        toc = ((double)cvGetTickCount() - t)/cvGetTickFrequency();
        toc = toc/1000000;
        float fps = 1 /toc;
        sprintf(strFrame, "%lf ",fps);
        putText(frame,strFrame,cvPoint(0,20),2,1,CV_RGB(25,200,25));
        imshow("TLD", frame);

        //swap points and images
        swap(last_gray,current_gray);
        pts1.clear();
        pts2.clear();
        frames++;
       // printf("Detection rate: %d/%d\n",detections,frames);
        if (cvWaitKey(33) == 'q')
            break;
    }
    if (rep){
        rep = false;
        tl = false;
        fclose(bb_file);
        bb_file = fopen("final_detector.txt","w");
        //capture.set(CV_CAP_PROP_POS_AVI_RATIO,0);
        capture.release();
        capture.open(video);
        goto REPEAT;
    }
    fclose(bb_file);
    return 0;
}
