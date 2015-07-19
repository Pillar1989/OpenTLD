/*
 * TLD.cpp
 *
 *  Created on: Jun 9, 2011
 *      Author: alantrrs
 */

#include "include/TLD.h"
#include <stdio.h>
#include <QDebug>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <algorithm>
#include <iterator>
#include<dirent.h>
using namespace cv;
using namespace std;


TLD::TLD()
{
}
TLD::TLD(const FileNode& file){
    read(file);
}

void TLD::read(const FileNode& file){
    ///Bounding Box Parameters
    min_win = (int)file["min_win"];
    qDebug() << "read min_win" << min_win;
    ///Genarator Parameters
    //initial parameters for positive examples
    patch_size = (int)file["patch_size"];
    qDebug() << "read patchsize" << patch_size;
    num_closest_init = (int)file["num_closest_init"];
    num_warps_init = (int)file["num_warps_init"];
    noise_init = (int)file["noise_init"];
    angle_init = (float)file["angle_init"];
    shift_init = (float)file["shift_init"];
    scale_init = (float)file["scale_init"];
    //update parameters for positive examples
    num_closest_update = (int)file["num_closest_update"];
    num_warps_update = (int)file["num_warps_update"];
    noise_update = (int)file["noise_update"];
    angle_update = (float)file["angle_update"];
    shift_update = (float)file["shift_update"];
    scale_update = (float)file["scale_update"];
    //parameters for negative examples
    bad_overlap = (float)file["overlap"];
    bad_patches = (int)file["num_patches"];
    classifier.read(file);
}

void TLD::init(const Mat& frame1,const Rect& box,FILE* bb_file){
    //bb_file = fopen("bounding_boxes.txt","w");
    //Get Bounding Boxes
    buildGrid(frame1,box);

    printf("Created %d bounding boxes\n",(int)grid.size());

    //Preparation
    //allocation
    //积分图像，用以计算2bitBP特征（类似于haar特征的计算）  
    //Mat的创建，方式有两种：1.调用create（行，列，类型）2.Mat（行，列，类型（值））。  
    iisum.create(frame1.rows+1,frame1.cols+1,CV_32F);
    iisqsum.create(frame1.rows+1,frame1.cols+1,CV_64F);
    //检查确信度
    dconf.reserve(100);
    //检测到物体的位子Detected BoundingBox
    dbb.reserve(100);
    //没有用到
    bbox_step =7;
    //Detector data
    //tmp.conf.reserve(grid.size());
    tmp.conf = vector<float>(grid.size());
    tmp.patt = vector<vector<int> >(grid.size(),vector<int>(10,0));
    //tmp.patt.reserve(grid.size());
    dt.bb.reserve(grid.size());
    good_boxes.reserve(grid.size());
    bad_boxes.reserve(grid.size());

    //TLD中定义：cv::Mat pEx;  //positive NN example 大小为15*15图像片
    //
    pEx = vector<Mat>(1,Mat(patch_size,patch_size,CV_64F));
    //pEx[0].create(patch_size,patch_size,CV_64F);

    //Init Generator
    generator = PatchGenerator (0,0,noise_init,true,1-scale_init,1+scale_init,-angle_init*CV_PI/180,angle_init*CV_PI/180,-angle_init*CV_PI/180,angle_init*CV_PI/180);
    //此函数根据传入的box（目标边界框），在整帧图像中的全部窗口中寻找与该box距离最小（即最相似，
    //重叠度最大）的num_closest_init个窗口，然后把这些窗口 归入good_boxes容器
    //同时，把重叠度小于0.2的，归入 bad_boxes 容器
    //首先根据overlap的比例信息选出重复区域比例大于60%并且前num_closet_init= 10个的最接近box的RectBox，
    //相当于对RectBox进行筛选。并通过BBhull函数得到这些RectBox的最大边界。
    getOverlappingBoxes(box,num_closest_init);
    printf("Found %d good boxes, %d bad boxes\n",(int)good_boxes.size(),(int)bad_boxes.size());
    printf("Best Box: %d %d %d %d\n",best_box.x,best_box.y,best_box.width,best_box.height);
    printf("Bounding box hull: %d %d %d %d\n",bbhull.x,bbhull.y,bbhull.width,bbhull.height);

    //Correct Bounding Box
    lastbox=best_box;
    lastconf=1;
    lastvalid=true;
    //Print
    fprintf(bb_file,"%d,%d,%d,%d,%f\n",lastbox.x,lastbox.y,lastbox.br().x,lastbox.br().y,lastconf);

    //Prepare Classifier 准备分类器
     //scales容器里是所有扫描窗口的尺度，由buildGrid()函数初始化
    classifier.prepare(scales);
    ///Generate Data
    // Generate positive data
    //产生P数据
    generatePositiveData(frame1,num_warps_init);

    // Set variance threshold
    Scalar stdev, mean;
    //计算方差和标准差
    meanStdDev(frame1(best_box),mean,stdev);
    //iisum,积分图
    //iisqsum,平方积分图
    integral(frame1,iisum,iisqsum);

    var = pow(stdev.val[0],2)*0.5; //getVar(best_box,iisum,iisqsum);
    cout << "variance: " << var << endl;//方差
    //check variance
    //getVar函数通过积分图像计算输入的best_box的方差
    double vr =  getVar(best_box,iisum,iisqsum)*0.5;
    cout << "check variance: " << vr << endl;
    // Generate negative data
    //经过正负样品的产生以后，分别产生了正样本pX和负样本nX， 以及正专家pEx和负专家nEx
    generateNegativeData(frame1);
    //Split Negative Ferns into Training and Testing sets (they are already shuffled)
    //将负样本放进 训练和测试集
    int half = (int)nX.size()*0.5f;
    //vector::assign函数将区间[start, end)中的值赋值给当前的vector.
    //将一半的负样本集 作为 测试集
    nXT.assign(nX.begin()+half,nX.end());//负样本测试集
    nX.resize(half);//负样本为原来一半
    //Split Negative NN Examples into Training and Testing sets
    //然后将剩下的一半作为训练集
    half = (int)nEx.size()*0.5f;
    nExT.assign(nEx.begin()+half,nEx.end());//负专家训练集
    nEx.resize(half);//负专家为原来一半
    //Merge Negative Data with Positive Data and shuffle it
     //将负样本和正样本合并，然后打乱
    vector<pair<vector<int>,int> > ferns_data(nX.size()+pX.size());
    //产生0到ferns_data.size()之间的随机数组，存到idx里面
    vector<int> idx = index_shuffle(0,ferns_data.size());
    int a=0;

    //把pX和nX的值随机存到ferns_data中
    for (int i=0;i<pX.size();i++){
        ferns_data[idx[a]] = pX[i];
        a++;
    }
    for (int i=0;i<nX.size();i++){
        ferns_data[idx[a]] = nX[i];
        a++;
    }

#ifdef AUTO
    int count;
    char name[64];
    FileStorage fs("D:\\opencv_project\\OpenTLD\\data\\PN_data.yml", FileStorage::READ);
    fs["pExtotal"] >> count;
    pEx_num = count;
    for(int i = 0;i < count;i++)
    {
         Mat img;
        sprintf(name,"pEx%d",i);
        fs[name] >> img;
        pEx.push_back(img);
    }
    fs["nExtotal"] >> count;
    for(int i = 0;i < count;i++)
    {
         Mat img;
        sprintf(name,"nEx%d",i);
        fs[name] >> img;
        nEx.push_back(img);
    }
#endif
     //把正专家和负专家放到一起
     //Data already have been shuffled, just putting it in the same vector
     vector<cv::Mat> nn_data(pEx.size());
     nn_data = pEx;
 //    nn_data[0] = pEx;
 //    for (int i=0;i<nEx.size();i++){
 //        nn_data[i+1]= nEx[i];
 //    }
     copy(nEx.begin(),nEx.end(),back_inserter(nn_data));
    ///Training
    //训练 集合分类器（森林） 和 最近邻分类器
    classifier.trainF(ferns_data,2); //bootstrap = 2，训练正负样本
    classifier.trainNN(nn_data,pEx_num);//训练正负专家
#ifdef AUTO
    classifier.addNN(nn_data,pEx_num);//添加正负专家数据
#endif
    ///Threshold Evaluation on testing sets
    ///用样本在上面得到的 集合分类器（森林） 和 最近邻分类器 中分类，评价得到最好的阈值
    classifier.evaluateTh(nXT,nExT);//评估
    cout << "init end ............" << endl;
    classifier.show();
}

/* Generate Positive data
 * Inputs:
 * - good_boxes (bbP)
 * - best_box (bbP0)
 * - frame (im0)
 * Outputs:
 * - Positive fern features (pX)
 * - Positive NN examples (pEx)
 */
void TLD::generatePositiveData(const Mat& frame, int num_warps){
    Scalar mean;//均值
    Scalar stdev;//标准差
    cv::Mat pEx_tmp(patch_size,patch_size,CV_64F);

    //此函数将frame图像best_box区域的图像片归一化为均值为0的15*15大小的patch，存在pEx正样本中
    getPattern(frame(best_box),pEx_tmp,mean,stdev);
    swap(pEx_tmp,pEx[0]);
    //pEx.push_back(pEx_tmp);
    pEx_num = 1;

    //Get Fern features on warped patches
    Mat img;
    Mat warped;
    GaussianBlur(frame,img,Size(9,9),1.5);
    //在img图像中截取bbhull信息（bbhull是包含了位置和大小的矩形框）的图像赋给warped
    //例如需要提取图像A的某个ROI（感兴趣区域，由矩形框）的话，用Mat类的B=img(ROI)即可提取
    warped = img(bbhull);
    RNG& rng = theRNG();
    //取矩形框中心的坐标  int i(2)
    Point2f pt(bbhull.x+(bbhull.width-1)*0.5f,bbhull.y+(bbhull.height-1)*0.5f);
    //nstructs树木（由一个特征组构建，每组特征代表图像块的不同视图表示）的个数
     //fern[nstructs] nstructs棵树的森林的数组？？
    vector<int> fern(classifier.getNumStructs());
    pX.clear();
    Mat patch;
     //pX为处理后的RectBox最大边界处理后的像素信息，pEx最近邻的RectBox的Pattern，bbP0为最近邻的RectBox。
    if (pX.capacity()<num_warps*good_boxes.size())
        //pX正样本个数为 仿射变换个数 * good_box的个数，故需分配至少这么大的空间
        pX.reserve(num_warps*good_boxes.size());
    int idx;
    for (int i=0;i<num_warps;i++){
        if (i>0)
            //generator(frame,pt,warped,bbhull.size(),rng);
             //PatchGenerator类用来对图像区域进行仿射变换，先RNG一个随机因子，再调用（）运算符产生一个变换后的正样本。
           // generator(img,pt,img,frame.size(),rng);
        for (int b=0;b<good_boxes.size();b++){
            //good_boxes容器保存的是 grid 的索引
            idx=good_boxes[b];
            //把img的 grid[idx] 区域（也就是bounding box重叠度高的）这一块图像片提取出来
            patch = img(grid[idx]);
            //getFeatures函数得到输入的patch的用于树的节点，也就是特征组的特征fern（13位的二进制代码）
              //grid[idx].sidx 对应的尺度索引
            classifier.getFeatures(patch,grid[idx].sidx,fern);
            //positive ferns <features, labels=1>  正样本，标签为1
            pX.push_back(make_pair(fern,1));
        }
    }
    printf("Positive examples generated: ferns:%d NN:1\n",(int)pX.size());
}

void TLD::getPattern(const Mat& img, Mat& pattern,Scalar& mean,Scalar& stdev){
    //newdata = (olddata - mindata)/(maxdata - mindata)
    //Output: resized Zero-Mean patch
    resize(img,pattern,Size(patch_size,patch_size));
    //计算均值和标准差
    meanStdDev(pattern,mean,stdev);
    pattern.convertTo(pattern,CV_32F);
    pattern = pattern-mean.val[0];
}

void TLD::generateNegativeData(const Mat& frame){
    /* Inputs:
 * - Image
 * - bad_boxes (Boxes far from the bounding box)
 * - variance (pEx variance)
 * Outputs
 * - Negative fern features (nX)
 * - Negative NN examples (nEx)
 */
    //由于之前重叠度小于0.2的，都归入 bad_boxes了，所以数量挺多，下面的函数用于打乱顺序，也就是为了
    //后面随机选择bad_boxes
    random_shuffle(bad_boxes.begin(),bad_boxes.end());//Random shuffle bad_boxes indexes
    int idx;
    //Get Fern Features of the boxes with big variance (calculated using integral images)
    int a=0;
    //int num = std::min((int)bad_boxes.size(),(int)bad_patches*100); //limits the size of bad_boxes to try
    cout << "negative data generation started." << endl;
    vector<int> fern(classifier.getNumStructs());
    nX.reserve(bad_boxes.size());
    Mat patch;
    for (int j=0;j<bad_boxes.size();j++){//把方差较大的bad_boxes加入负样本
        idx = bad_boxes[j];
        if (getVar(grid[idx],iisum,iisqsum)<var*0.5f)
            continue;
        patch =  frame(grid[idx]);
        classifier.getFeatures(patch,grid[idx].sidx,fern);
        //负样本标签为0
        nX.push_back(make_pair(fern,0));
        a++;
    }
    cout << "Negative examples generated: ferns: " << a << endl;
    //random_shuffle(bad_boxes.begin(),bad_boxes.begin()+bad_patches);//Randomly selects 'bad_patches' and get the patterns for NN;
    Scalar dum1, dum2;
    nEx=vector<Mat>(bad_patches);
    for (int i=0;i<bad_patches;i++){
        idx=bad_boxes[i];
        patch = frame(grid[idx]);
        //具体的说就是归一化RectBox对应的patch的size（放缩至patch_size = 15*15）
        //由于负样本不需要均值和方差，所以就定义dum，将其舍弃
        getPattern(patch,nEx[i],dum1,dum2);
    }
    cout << "NN: " << (int)nEx.size() << endl;
}
//该函数通过积分图像计算输入的box的方差
double TLD::getVar(const BoundingBox& box,const Mat& sum,const Mat& sqsum){
    double brs = sum.at<int>(box.y+box.height,box.x+box.width);
    double bls = sum.at<int>(box.y+box.height,box.x);
    double trs = sum.at<int>(box.y,box.x+box.width);
    double tls = sum.at<int>(box.y,box.x);
    double brsq = sqsum.at<double>(box.y+box.height,box.x+box.width);
    double blsq = sqsum.at<double>(box.y+box.height,box.x);
    double trsq = sqsum.at<double>(box.y,box.x+box.width);
    double tlsq = sqsum.at<double>(box.y,box.x);
    double mean = (brs+tls-trs-bls)/((double)box.area());
    double sqmean = (brsq+tlsq-trsq-blsq)/((double)box.area());
    //方差=E(X^2)-(EX)^2   EX表示均值
    return sqmean-mean*mean;
}

void TLD::processFrame(const cv::Mat& img1,const cv::Mat& img2,vector<Point2f>& points1,
                       vector<Point2f>& points2,BoundingBox& bbnext,bool& lastboxfound,
                       bool tl, FILE* bb_file){
    vector<BoundingBox> cbb;
    vector<float> cconf;
    int confident_detections=0;//小D的结果聚类之后，分数比小T高的数目
    int didx; //detection index

    ///Track
    if(lastboxfound && tl){//前一帧目标出现过，我们才跟踪，否则只能检测了
        track(img1,img2,points1,points2);
    }
    else{
        tracked = false;
    }
    ///Detect
    detect(img2);

    ///Integration
    //TLD只跟踪单目标，所以综合模块综合跟踪器跟踪到的单个目标和检测器检测到的多个目标，然后只输出保守相似度最大的一个目标
    if (tracked){
        bbnext=tbb;
        lastconf=tconf; //表示相关相似度的阈值
        lastvalid=tvalid;
        printf("Tracked\n");
        if(detected){                                               //   if Detected
            //通过 重叠度 对检测器检测到的目标bounding box进行聚类，每个类其重叠度小于0.5
            clusterConf(dbb,dconf,cbb,cconf);                       //   cluster detections

#if 0
        cv::Mat img = img2.clone();
        for (int i=0;i<cbb.size();i++){
              drawBox(img,cbb[i]);
          }
        imshow("detections",img);
#endif
            //printf("Found %d clusters\n",(int)cbb.size());
            for (int i=0;i<cbb.size();i++){
                 //找到与跟踪器跟踪到的box距离比较远的类（检测器检测到的box），而且它的相关相似度比跟踪器的要大
                if (bbOverlap(tbb,cbb[i])<0.5 && cconf[i]>tconf){  //  Get index of a clusters that is far from tracker and are more confident than the tracker
                    confident_detections++;
                    didx=i; //detection index
                }
            }
            //如果只有一个满足上述条件的box，那么就用这个目标box来重新初始化跟踪器（也就是用检测器的结果去纠正跟踪器）
            if (confident_detections==1){                                //if there is ONE such a cluster, re-initialize the tracker
                //printf("Found a better match..reinitializing tracking\n");
                bbnext=cbb[didx];
                lastconf=cconf[didx];
                lastvalid=false;
            }
            else {
                //printf("%d confident cluster was found\n",confident_detections);
                int cx=0,cy=0,cw=0,ch=0;
                int close_detections=0;
                for (int i=0;i<dbb.size();i++){
                    //找到检测器检测到的box与跟踪器预测到的box距离很近（重叠度大于0.7）的box，对其坐标和大小进行累加
                    if(bbOverlap(tbb,dbb[i])>0.7){                     // Get mean of close detections
                        cx += dbb[i].x;
                        cy +=dbb[i].y;
                        cw += dbb[i].width;
                        ch += dbb[i].height;
                        close_detections++;//记录最近邻box的个数
                        //printf("weighted detection: %d %d %d %d\n",dbb[i].x,dbb[i].y,dbb[i].width,dbb[i].height);
                    }
                }
                if (close_detections>0){
                    //对与跟踪器预测到的box距离很近的box 和 跟踪器本身预测到的box 进行坐标与大小的平均作为最终的
                    //目标bounding box，但是跟踪器的权值较大
                    bbnext.x = cvRound((float)(10*tbb.x+cx)/(float)(10+close_detections));   // weighted average trackers trajectory with the close detections
                    bbnext.y = cvRound((float)(10*tbb.y+cy)/(float)(10+close_detections));
                    bbnext.width = cvRound((float)(10*tbb.width+cw)/(float)(10+close_detections));
                    bbnext.height =  cvRound((float)(10*tbb.height+ch)/(float)(10+close_detections));
                    //printf("Tracker bb: %d %d %d %d\n",tbb.x,tbb.y,tbb.width,tbb.height);
                   // printf("Average bb: %d %d %d %d\n",bbnext.x,bbnext.y,bbnext.width,bbnext.height);
                    //printf("Weighting %d close detection(s) with tracker..\n",close_detections);
                }
                else{
                    //printf("%d close detections were found\n",close_detections);

                }
            }
        }
    }
    else{                                       //   If NOT tracking
       // printf("Not tracking..\n");
        lastboxfound = false;
        lastvalid = false;
        //如果跟踪器没有跟踪到目标，但是检测器检测到了一些可能的目标box，那么同样对其进行聚类，但只是简单的
        //将聚类的cbb[0]作为新的跟踪目标box（不比较相似度了？？还是里面已经排好序了？？），重新初始化跟踪器
        if(detected){                           //  and detector is defined
            clusterConf(dbb,dconf,cbb,cconf);   //  cluster detections
           // printf("Found %d clusters\n",(int)cbb.size());
            if (cconf.size()==1){
                bbnext=cbb[0];
                lastconf=cconf[0];
                //printf("Confident detection..reinitializing tracker\n");
                lastboxfound = true;

            }
        }
    }
    lastbox=bbnext;
    if (lastboxfound)
        fprintf(bb_file,"%d,%d,%d,%d,%f\n",lastbox.x,lastbox.y,lastbox.br().x,lastbox.br().y,lastconf);
    else
        fprintf(bb_file,"NaN,NaN,NaN,NaN,NaN\n");
    if (lastvalid && tl)
        learn(img2);
}

//points1网点，points2光流点
void TLD::track(const Mat& img1, const Mat& img2,vector<Point2f>& points1,vector<Point2f>& points2){
    /*Inputs:
   * -current frame(img2), last frame(img1), last Bbox(bbox_f[0]).
   *Outputs:
   *- Confidence(tconf), Predicted bounding box(tbb),Validity(tvalid), points2 (for display purposes only)
   */
    //Generate points
    //网格均匀撒点（均匀采样），在lastbox中共产生最多10*10=100个特征点，存于points1
    bbPoints(points1,lastbox);

    if (points1.size()<1){
        printf("BB= %d %d %d %d, Points not generated\n",lastbox.x,lastbox.y,lastbox.width,lastbox.height);
        tvalid=false;
        tracked=false;
        return;
    }
    vector<Point2f> points = points1;
    //Frame-to-frame tracking with forward-backward error cheking
    //trackf2f函数完成：跟踪、计算FB error和匹配相似度sim，然后筛选出 FB_error[i] <= median(FB_error) 和
    //sim_error[i] > median(sim_error) 的特征点（跟踪结果不好的特征点），剩下的是不到50%的特征点
    tracked = tracker.trackf2f(img1,img2,points,points2);

    if (tracked){
        //Bounding box prediction
        //利用剩下的这不到一半的跟踪点输入来预测bounding box在当前帧的位置和大小 tbb
        //跟踪失败检测：如果FB error的中值大于10个像素（经验值），或者预测到的当前box的位置移出图像，则
        //认为跟踪错误，此时不返回bounding box；Rect::br()返回的是右下角的坐标
        //getFB()返回的是FB error的中值
        bbPredict(points,points2,lastbox,tbb);
        if (tracker.getFB()>10 || tbb.x>img2.cols ||  tbb.y>img2.rows || tbb.br().x < 1 || tbb.br().y <1){
            tvalid =false; //too unstable prediction or bounding box out of image
            tracked = false;
            printf("Too unstable predictions FB error=%f\n",tracker.getFB());
            return;
        }
        //Estimate Confidence and Validity
        //评估跟踪确信度和有效性
        Mat pattern;
        Scalar mean, stdev;
        BoundingBox bb;
        bb.x = max(tbb.x,0);
        bb.y = max(tbb.y,0);
        bb.width = min(min(img2.cols-tbb.x,tbb.width),min(tbb.width,tbb.br().x));
        bb.height = min(min(img2.rows-tbb.y,tbb.height),min(tbb.height,tbb.br().y));
        //归一化img2(bb)对应的patch的size（放缩至patch_size = 15*15），存入pattern
        getPattern(img2(bb),pattern,mean,stdev);   
        vector<int> isin;
        float dummy;
        //计算图像片pattern到在线模型M的保守相似度
        classifier.NNConf(pattern,isin,dummy,tconf); //Conservative Similarity

        tvalid = lastvalid;
        //保守相似度大于阈值，则评估跟踪有效
        if (tconf>classifier.thr_nn_valid){
            tvalid =true;
        }


    }
    else
        printf("No points tracked\n");

}
//产生网点
void TLD::bbPoints(vector<cv::Point2f>& points,const BoundingBox& bb){
    int max_pts=10;
    int margin_h=0;
    int margin_v=0;
    int stepx = ceil((bb.width-2*margin_h)/max_pts);
    int stepy = ceil((bb.height-2*margin_v)/max_pts);
    for (int y=bb.y+margin_v;y<bb.y+bb.height-margin_v;y+=stepy){
        for (int x=bb.x+margin_h;x<bb.x+bb.width-margin_h;x+=stepx){
            points.push_back(Point2f(x,y));
        }
    }
}
//利用剩下的这不到一半的跟踪点输入来预测bounding box在当前帧的位置和大小
void TLD::bbPredict(const vector<cv::Point2f>& points1,const vector<cv::Point2f>& points2,
                    const BoundingBox& bb1,BoundingBox& bb2)    {
    int npoints = (int)points1.size();
    vector<float> xoff(npoints);//位移
    vector<float> yoff(npoints);
    printf("tracked points : %d\n",npoints);
    for (int i=0;i<npoints;i++){//计算每个特征点在两帧之间的位移
        xoff[i]=points2[i].x-points1[i].x;
        yoff[i]=points2[i].y-points1[i].y;
    }
    float dx = median(xoff); //计算位移的中值
    float dy = median(yoff);
    float s;
    //计算bounding box尺度scale的变化：通过计算 当前特征点相互间的距离 与 先前（上一帧）特征点相互间的距离 的
    //比值，以比值的中值作为尺度的变化因子
    if (npoints>1){
        vector<float> d;
        d.reserve(npoints*(npoints-1)/2);//等差数列求和：1+2+...+(npoints-1)
        for (int i=0;i<npoints;i++){
            for (int j=i+1;j<npoints;j++){
                //计算 当前特征点相互间的距离 与 先前（上一帧）特征点相互间的距离 的比值（位移用绝对值）
                //之前比较的都是对应点之间的相似性，现在计算的是任意两点的相似性，所以更能反映拓扑结构的变化
                d.push_back(norm(points2[i]-points2[j])/norm(points1[i]-points1[j]));
            }
        }
        s = median(d);
    }
    else {
        s = 1.0;
    }
    //应该是 _______     不应该是  _______
    //     |\_____ |           |     | |
    //     ||     ||           |     | |
    //     ||  1  ||           |  2  | |
    //     ||     ||           |-----  |
    //     | ----- |           |_______|
    //      -------
    //如果没有s1,s1出现的图形就是2
    float s1 = 0.5*(s-1)*bb1.width;// top-left 坐标的偏移(s1,s2)
    float s2 = 0.5*(s-1)*bb1.height;
    //printf("s= %f s1= %f s2= %f ....................... \n",s,s1,s2);
    //得到当前bounding box的位置与大小信息
    //当前box的x坐标 = 前一帧box的x坐标 + 全部特征点位移的中值（可理解为box移动近似的位移） - 当前box宽的一半
    bb2.x = round( bb1.x + dx -s1);
    bb2.y = round( bb1.y + dy -s2);

    bb2.width = round(bb1.width*s);
    bb2.height = round(bb1.height*s);
   // printf("predicted bb: %d %d %d %d\n",bb2.x,bb2.y,bb2.br().x,bb2.br().y);
}

void TLD::detect(const cv::Mat& frame){
    //cleaning
    dbb.clear();
    dconf.clear();
    dt.bb.clear();
     //GetTickCount返回从操作系统启动到现在所经过的时间
    double t = (double)getTickCount();
    Mat img(frame.rows,frame.cols,CV_8U);
    integral(frame,iisum,iisqsum);//计算frame的积分图
    GaussianBlur(frame,img,Size(9,9),1.5); //高斯模糊
    int numtrees = classifier.getNumStructs();
    float fern_th = classifier.getFernTh(); //getFernTh()返回thr_fern(0.6); 集合分类器的分类阈值
    vector <int> ferns(10);
    float conf;
    int a=0;
    Mat patch;
    //级联分类器模块一：方差检测模块，利用积分图计算每个待检测窗口的方差，方差大于var阈值（目标patch方差的50%）的，
     //则认为其含有前景目标
    for (int i=0;i<grid.size();i++){//FIXME: BottleNeck瓶颈
        if (getVar(grid[i],iisum,iisqsum)>=var){//计算每一个扫描窗口的方差
            a++;
            //级联分类器模块二：集合分类器检测模块
            patch = img(grid[i]);
            classifier.getFeatures(patch,grid[i].sidx,ferns); //得到该patch特征（13位的二进制代码）
            conf = classifier.measure_forest(ferns);//计算该特征值对应的后验概率累加值
            tmp.conf[i]=conf; //Detector data中定义TempStruct tmp;
            tmp.patt[i]=ferns;
            //如果集合分类器的后验概率的平均值大于阈值fern_th（由训练得到），就认为含有前景目标
            if (conf>numtrees*fern_th){
                dt.bb.push_back(i);//将通过以上两个检测模块的扫描窗口记录在detect structure中
            }
        }
        else
            tmp.conf[i]=0.0;
    }
    int detections = dt.bb.size();
    //printf("%d Bounding boxes passed the variance filter\n",a);
    //cout << "Initial detection from Fern Classifier:" << detections << endl;
    //如果通过以上两个检测模块的扫描窗口数大于100个，则只取后验概率大的前100个
    if (detections>100){ //CComparator(tmp.conf)指定比较方式？？？
        nth_element(dt.bb.begin(),dt.bb.begin()+100,dt.bb.end(),CComparator(tmp.conf));
        dt.bb.resize(100);
        detections=100;
    }
    #if 0
      for (int i=0;i<detections;i++){
            drawBox(img,grid[dt.bb[i]]);
        }
      imshow("detections",img);
    #endif
    if (detections == 0){
        detected=false;
        return;
    }
    //printf("Fern detector made %d detections ",detections);
     //两次使用getTickCount()，然后再除以getTickFrequency()，计算出来的是以秒s为单位的时间（opencv 2.0 以前是ms）
    t=(double)getTickCount()-t;
   // printf("in %gms\n", t*1000/getTickFrequency());
    //  Initialize detection structure
    dt.patt = vector<vector<int> >(detections,vector<int>(10,0));        //  Corresponding codes of the Ensemble Classifier
    dt.conf1 = vector<float>(detections);                                //  Relative Similarity (for final nearest neighbour classifier)
    dt.conf2 =vector<float>(detections);                                 //  Conservative Similarity (for integration with tracker)
    dt.isin = vector<vector<int> >(detections,vector<int>(3,-1));        //  Detected (isin=1) or rejected (isin=0) by nearest neighbour classifier
    dt.patch = vector<Mat>(detections,Mat(patch_size,patch_size,CV_32F));//  Corresponding patches
    int idx;
    Scalar mean, stdev;
    float nn_th = classifier.getNNTh();
     //级联分类器模块三：最近邻分类器检测模块
    for (int i=0;i<detections;i++){                                         //  for every remaining detection
        idx=dt.bb[i];                                                       //  Get the detected bounding box index
        patch = frame(grid[idx]);
        getPattern(patch,dt.patch[i],mean,stdev);                //  Get pattern within bounding box
        //计算图像片pattern到在线模型M的相关相似度和保守相似度
        classifier.NNConf(dt.patch[i],dt.isin[i],dt.conf1[i],dt.conf2[i]);  //  Evaluate nearest neighbour classifier
        dt.patt[i]=tmp.patt[idx];
        //printf("Testing feature %d, conf:%f isin:(%d|%d|%d)\n",i,dt.conf1[i],dt.isin[i][0],dt.isin[i][1],dt.isin[i][2]);

        //相关相似度大于阈值，则认为含有前景目标
        if (dt.conf1[i]>nn_th){                                               //  idx = dt.conf1 > tld.model.thr_nn; % get all indexes that made it through the nearest neighbour
            dbb.push_back(grid[idx]);                                         //  BB    = dt.bb(:,idx); % bounding boxes
            dconf.push_back(dt.conf2[i]);                                     //  Conf  = dt.conf2(:,idx); % conservative confidences
            cout << "conf:  " << dt.conf1[i] << endl;
        }
    }                                                                         //  end
      //打印检测到的可能存在目标的扫描窗口数（可以通过三个级联检测器的）
    if (dbb.size()>0){
       // printf("Found %d NN matches\n",(int)dbb.size());
        detected=true;
#if 1
        for (int i=0;i<dbb.size();i++){
              drawBox(img,dbb[i]);
          }
        imshow("detections",img);
#endif
    }
    else{
        printf("No NN matches found.\n");
        detected=false;
    }
}

void TLD::evaluate(){
}

void TLD::learn(const Mat& img){
    printf("[Learning] ");
    ///Check consistency
    //检测一致性
    BoundingBox bb;
    bb.x = max(lastbox.x,0);
    bb.y = max(lastbox.y,0);
    bb.width = min(min(img.cols-lastbox.x,lastbox.width),min(lastbox.width,lastbox.br().x));
    bb.height = min(min(img.rows-lastbox.y,lastbox.height),min(lastbox.height,lastbox.br().y));
    Scalar mean, stdev;
    Mat pattern;
    //归一化img(bb)对应的patch的size（放缩至patch_size = 15*15），存入pattern
    getPattern(img(bb),pattern,mean,stdev);
    vector<int> isin;
    float dummy, conf;
    //计算输入图像片（跟踪器的目标box）与在线模型之间的相关相似度conf
    classifier.NNConf(pattern,isin,conf,dummy);
    if (conf<0.5) { //如果相似度太小了，就不训练
        //printf("Fast change..not training\n");
        lastvalid =false;
        return;
    }
    if (pow(stdev.val[0],2)<var){//如果方差太小了，也不训练
        //printf("Low variance..not training\n");
        lastvalid=false;
        return;
    }
    if(isin[2]==1){//如果被被识别为负样本，也不训练
        printf("Patch in negative data..not traing");
        lastvalid=false;
        return;
    }
    /// Data generation  样本产生
    for (int i=0;i<grid.size();i++){//计算所有的扫描窗口与目标box的重叠度
        grid[i].overlap = bbOverlap(lastbox,grid[i]);//更新重叠度
    }
     //集合分类器
    vector<pair<vector<int>,int> > fern_examples;
    good_boxes.clear();
    bad_boxes.clear();
    //此函数根据传入的lastbox，在整帧图像中的全部窗口中寻找与该lastbox距离最小（即最相似，
    //重叠度最大）的num_closest_update个窗口，然后把这些窗口 归入good_boxes容器（只是把网格数组的索引存入）
    //同时，把重叠度小于0.2的，归入 bad_boxes 容器
    getOverlappingBoxes(lastbox,num_closest_update);
    if (good_boxes.size()>0)
        //用仿射模型产生正样本（类似于第一帧的方法，但只产生10*10=100个）
        generatePositiveData(img,num_warps_update);
    else{
        lastvalid = false;
        printf("No good boxes..Not training");
        return;
    }
    fern_examples.reserve(pX.size()+bad_boxes.size());
    fern_examples.assign(pX.begin(),pX.end());
    int idx;
    for (int i=0;i<bad_boxes.size();i++){
        idx=bad_boxes[i];
        if (tmp.conf[idx]>=1){ //加入负样本，相似度大于1？？相似度不是出于0和1之间吗？
            fern_examples.push_back(make_pair(tmp.patt[idx],0));//负样本，标签为0
        }
    }
    //最近邻分类器
    vector<Mat> nn_examples;
    vector<Mat> tmp_nn_examples;
    nn_examples.reserve(pEx.size());
    nn_examples = pEx;
    tmp_nn_examples.reserve(dt.bb.size());
    //nn_examples.push_back(pEx);
    for (int i=0;i<dt.bb.size();i++){
        idx = dt.bb[i];
        if (bbOverlap(lastbox,grid[idx]) < bad_overlap)
            tmp_nn_examples.push_back(dt.patch[i]);
    }
    //把负样本都放到正样本的后面
    copy(tmp_nn_examples.begin(),tmp_nn_examples.end(),back_inserter(nn_examples));
    /// Classifiers update  分类器训练
    classifier.trainF(fern_examples,2);
    classifier.trainNN(nn_examples,pEx.size());
    //把正样本库（在线模型）包含的所有正样本显示在窗口上
    classifier.show();
#ifndef AUTO
    classifier.storeNP(true);
#else
    classifier.storeNP(false);
#endif
}
void TLD::drawGrid(cv::Mat &img)
{
    int i;
    cv::Rect rect;
    qDebug() << "good_boxes.size:  " << good_boxes.size();
    for(i = 0; i < good_boxes.size();i++)
    {
        rect.x = grid[good_boxes[i]].x;
        rect.y = grid[good_boxes[i]].y;
        rect.width = grid[good_boxes[i]].width;
        rect.height = grid[good_boxes[i]].height;
        rectangle(img,rect,Scalar(255,good_boxes[i]%255,0));
    }
    rectangle(img,bbhull,Scalar(255,good_boxes[i]%255,0),3);
}
/*1，先利用基数为0.16151的缩放倍数，以1.2的倍数建造缩放数字，预备后期对矩形框进行21次缩放，
缩放以后大小如何符合要求就放入scales中
   2，对每一次缩放的矩阵以10%的步长在图像中扫描，得到图像元，放入grid中
*/
void TLD::buildGrid(const cv::Mat& img, const cv::Rect& box){
    /*扫描窗口步长为 宽高的 10%*/
    const float SHIFT = 0.1;
    /*尺度缩放系数为1.2 （eg,0.16151*1.2=0.19381），共21种尺度变换 */
    const float SCALES[] = {0.16151,0.19381,0.23257,0.27908,0.33490,0.40188,0.48225,
                                       0.57870,0.69444,0.83333,1,1.20000,1.44000,1.72800,
                                       2.07360,2.48832,2.98598,3.58318,4.29982,5.15978,6.19174};
    int width, height, min_bb_side;
    //Rect bbox;
    BoundingBox bbox;
    Size scale;
    int sc=0;
    for (int s=0;s<21;s++){
        width = round(box.width*SCALES[s]);
        height = round(box.height*SCALES[s]);
        min_bb_side = min(height,width);    //bounding box最短的边
//由于图像片（min_win 为15x15像素）是在bounding box中采样得到的，所以box必须比min_win要大  
//另外，输入的图像肯定得比 bounding box 要大了 
        if (min_bb_side < min_win || width > img.cols || height > img.rows)
            continue;
        scale.width = width;
        scale.height = height;
        scales.push_back(scale);
        for (int y=1;y<img.rows-height;y+=round(SHIFT*min_bb_side)){
            for (int x=1;x<img.cols-width;x+=round(SHIFT*min_bb_side)){
                bbox.x = x;
                bbox.y = y;
                bbox.width = width;
                bbox.height = height;
                bbox.overlap = bbOverlap(bbox,BoundingBox(box)); //计算和原始矩形的重叠度
                bbox.sidx = sc;//缩放的索引
                grid.push_back(bbox);
            }
        }
        sc++;
    }
}
//此函数计算两个bounding box 的重叠度  
//重叠度定义为 两个box的交集 与 它们的并集 的比 
float TLD::bbOverlap(const BoundingBox& box1,const BoundingBox& box2){
    //先判断坐标，假如它们都没有重叠的地方，就直接返回0  
    if (box1.x > box2.x+box2.width) { return 0.0; }
    if (box1.y > box2.y+box2.height) { return 0.0; }
    if (box1.x+box1.width < box2.x) { return 0.0; }
    if (box1.y+box1.height < box2.y) { return 0.0; }
    //计算重叠区域的长和高
    float colInt =  min(box1.x+box1.width,box2.x+box2.width) - max(box1.x, box2.x);
    float rowInt =  min(box1.y+box1.height,box2.y+box2.height) - max(box1.y,box2.y); 
    //算出面积
    float intersection = colInt * rowInt;
    float area1 = box1.width*box1.height;
    float area2 = box2.width*box2.height;
    return intersection / (area1 + area2 - intersection);
}

void TLD::getOverlappingBoxes(const cv::Rect& box1,int num_closest){
    float max_overlap = 0;
    for (int i=0;i<grid.size();i++){
        if (grid[i].overlap > max_overlap) {
            max_overlap = grid[i].overlap;
            best_box = grid[i];
        }
        if (grid[i].overlap > 0.6){
            good_boxes.push_back(i);
        }
        else if (grid[i].overlap < bad_overlap){
            bad_boxes.push_back(i);
        }
    }
    //Get the best num_closest (10) boxes and puts them in good_boxes
    if (good_boxes.size()>num_closest){
        std::nth_element(good_boxes.begin(),good_boxes.begin()+num_closest,good_boxes.end(),OComparator(grid));
        good_boxes.resize(num_closest);
    }
    //找到最大的矩形
    getBBHull();
}

void TLD::getBBHull(){
    int x1=INT_MAX, x2=0;
    int y1=INT_MAX, y2=0;
    int idx;
    for (int i=0;i<good_boxes.size();i++){
        idx= good_boxes[i];
        x1=min(grid[idx].x,x1);
        y1=min(grid[idx].y,y1);
        x2=max(grid[idx].x+grid[idx].width,x2);
        y2=max(grid[idx].y+grid[idx].height,y2);
    }
    bbhull.x = x1;
    bbhull.y = y1;
    bbhull.width = x2-x1;
    bbhull.height = y2 -y1;
}

bool bbcomp(const BoundingBox& b1,const BoundingBox& b2){
    TLD t;
    if (t.bbOverlap(b1,b2)<0.5)
        return false;
    else
        return true;
}
int TLD::clusterBB(const vector<BoundingBox>& dbb,vector<int>& indexes){
    //FIXME: Conditional jump or move depends on uninitialised value(s)
    const int c = dbb.size();
    //1. Build proximity matrix
    Mat D(c,c,CV_32F);
    float d;
    for (int i=0;i<c;i++){
        for (int j=i+1;j<c;j++){
            d = 1-bbOverlap(dbb[i],dbb[j]);
            D.at<float>(i,j) = d;
            D.at<float>(j,i) = d;
        }
    }
    //2. Initialize disjoint clustering
    float L[c-1]; //Level
    int nodes[c-1][2];
    int belongs[c];
    int m=c;
    for (int i=0;i<c;i++){
        belongs[i]=i;
    }
    for (int it=0;it<c-1;it++){
        //3. Find nearest neighbor
        float min_d = 1;
        int node_a, node_b;
        for (int i=0;i<D.rows;i++){
            for (int j=i+1;j<D.cols;j++){
                if (D.at<float>(i,j)<min_d && belongs[i]!=belongs[j]){
                    min_d = D.at<float>(i,j);
                    node_a = i;
                    node_b = j;
                }
            }
        }
        if (min_d>0.5){
            int max_idx =0;
            bool visited;
            for (int j=0;j<c;j++){
                visited = false;
                for(int i=0;i<2*c-1;i++){
                    if (belongs[j]==i){
                        indexes[j]=max_idx;
                        visited = true;
                    }
                }
                if (visited)
                    max_idx++;
            }
            return max_idx;
        }

        //4. Merge clusters and assign level
        L[m]=min_d;
        nodes[it][0] = belongs[node_a];
        nodes[it][1] = belongs[node_b];
        for (int k=0;k<c;k++){
            if (belongs[k]==belongs[node_a] || belongs[k]==belongs[node_b])
                belongs[k]=m;
        }
        m++;
    }
    return 1;

}
//对检测器检测到的目标bounding box进行聚类
//聚类（Cluster）分析是由若干模式（Pattern）组成的，通常，模式是一个度量（Measurement）的向量，或者是多维空间中的
//一个点。聚类分析以相似性为基础，在一个聚类中的模式之间比不在同一聚类中的模式之间具有更多的相似性。
void TLD::clusterConf(const vector<BoundingBox>& dbb,const vector<float>& dconf,
                      vector<BoundingBox>& cbb,vector<float>& cconf){
    int numbb =dbb.size();
    vector<int> T;
    float space_thr = 0.5;
    int c=1;//记录 聚类的类个数
    switch (numbb){//检测到的含有目标的bounding box个数
    case 1:
        cbb=vector<BoundingBox>(1,dbb[0]);//如果只检测到一个，那么这个就是检测器检测到的目标
        cconf=vector<float>(1,dconf[0]);
        return;
        break;
    case 2:
        T =vector<int>(2,0);
         //此函数计算两个bounding box 的重叠度
        //如果只检测到两个box，但他们的重叠度小于0.5
        if (1-bbOverlap(dbb[0],dbb[1])>space_thr){
            T[1]=1;
            c=2; //重叠度小于0.5的box，属于不同的类
        }
        break;
    default: //重叠度小于0.5的box，属于不同的类
        T = vector<int>(numbb,0);
        //stable_partition()重新排列元素，使得满足指定条件的元素排在不满足条件的元素前面。它维持着两组元素的顺序关系。
        //STL partition就是把一个区间中的元素按照某个条件分成两类。返回第二类子集的起点
        //bbcomp()函数判断两个box的重叠度小于0.5，返回false，否则返回true （分界点是重叠度：0.5）
        //partition() 将dbb划分为两个子集，将满足两个box的重叠度小于0.5的元素移动到序列的前面，为一个子集，重叠度大于0.5的，
        //放在序列后面，为第二个子集，但两个子集的大小不知道，返回第二类子集的起点
        //重叠度小于0.5的box，属于不同的类，所以c是不同的类别个数
        c = partition(dbb,T,(*bbcomp));
        //c = clusterBB(dbb,T);
        break;
    }
    cconf=vector<float>(c);
    cbb=vector<BoundingBox>(c);
    printf("Cluster indexes ->c:%d T.size:%d ",c,T.size());

    BoundingBox bx;
    for (int i=0;i<c;i++){ //类别个数
        float cnf=0;
        int N=0,mx=0,my=0,mw=0,mh=0;
        for (int j=0;j<T.size();j++){//检测到的bounding box个数
            if (T[j]==i){//将聚类为同一个类别的box的坐标和大小进行累加
                printf("%d ",i);
                cnf=cnf+dconf[j];
                mx=mx+dbb[j].x;
                my=my+dbb[j].y;
                mw=mw+dbb[j].width;
                mh=mh+dbb[j].height;
                N++;
            }
        }
        if (N>0){ //然后求该类的box的坐标和大小的平均值，将平均值作为该类的box的代表
            cconf[i]=cnf/N;
            bx.x=cvRound(mx/N);
            bx.y=cvRound(my/N);
            bx.width=cvRound(mw/N);
            bx.height=cvRound(mh/N);
            cbb[i]=bx;//返回的是聚类，每一个类都有一个代表的bounding box
        }
    }
    printf("\n");
}

