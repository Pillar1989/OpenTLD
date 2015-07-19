/*
 * FerNNClassifier.cpp
 *
 *  Created on: Jun 14, 2011
 *      Author: alantrrs
 */

#include "include/FerNNClassifier.h"
#include <algorithm>
#include <iterator>
#include<dirent.h>
using namespace cv;
using namespace std;

void FerNNClassifier::read(const FileNode& file){
    ///Classifier Parameters
    valid = (float)file["valid"]; //0.5
    ncc_thesame = (float)file["ncc_thesame"]; //0.95
    nstructs = (int)file["num_trees"]; //10
    structSize = (int)file["num_features"]; //13
    thr_fern = (float)file["thr_fern"]; //0.6
    thr_nn = (float)file["thr_nn"]; //0.65
    thr_nn_valid = (float)file["thr_nn_valid"]; //0.7
}

//准备随机森林分类器
void FerNNClassifier::prepare(const vector<Size>& scales){
    acum = 0;
    //Initialize test locations for features
    int totalFeatures = nstructs*structSize;
    //二维向量  包含全部尺度（scales）的扫描窗口，每个尺度包含totalFeatures个特征
    features = vector<vector<Feature> >(scales.size(),vector<Feature> (totalFeatures));

    //opencv中自带的一个随机数发生器的类RNG
    RNG& rng = theRNG();
    float x1f,x2f,y1f,y2f;
    int x1, x2, y1, y2;

    //集合分类器基于n个基本分类器，每个分类器都是基于一个pixel comparisons（像素比较集）的；
    //pixel comparisons的产生方法：先用一个归一化的patch去离散化像素空间，产生所有可能的垂直和水平的pixel comparisons
    //然后我们把这些pixel comparisons随机分配给n个分类器，每个分类器得到完全不同的pixel comparisons（特征集合），
    //这样，所有分类器的特征组统一起来就可以覆盖整个patch了
    //用随机数去填充每一个尺度扫描窗口的特征
    for (int i=0;i<totalFeatures;i++){
        x1f = (float)rng;//产生[0,1)直接的浮点数  
        y1f = (float)rng;
        x2f = (float)rng;
        y2f = (float)rng;
        for (int s=0;s<scales.size();s++){
            x1 = x1f * scales[s].width;
            y1 = y1f * scales[s].height;

            x2 = x2f * scales[s].width;
            y2 = y2f * scales[s].height;
            //第s种尺度的第i个特征  两个随机分配的像素点坐标
            features[s][i] = Feature(x1, y1, x2, y2);
        }
    }
    //Thresholds
    thrN = 0.5*nstructs;

    //Initialize Posteriors  初始化后验概率
    //后验概率指每一个分类器对传入的图像片进行像素对比，每一个像素对比得到0或者1，所有的特征13个comparison对比，
    //连成一个13位的二进制代码x，然后索引到一个记录了后验概率的数组P(y|x)，y为0或者1（二分类），也就是出现x的
    //基础上，该图像片为y的概率是多少对n个基本分类器的后验概率做平均，大于0.5则判定其含有目标
    for (int i = 0; i<nstructs; i++) {
        //每一个每类器维护一个后验概率的分布，这个分布有2^d个条目（entries），这里d是像素比较pixel comparisons
        //的个数，这里是structSize，即13个comparison，所以会产生2^13即8,192个可能的code，每一个code对应一个后验概率
        //后验概率P(y|x)= #p/(#p+#n) ,#p和#n分别是正和负图像片的数目，也就是下面的pCounter和nCounter
        //初始化时，每个后验概率都得初始化为0；运行时候以下面方式更新：已知类别标签的样本（训练样本）通过n个分类器
        //进行分类，如果分类结果错误，那么响应的#p和#n就会更新，这样P(y|x)也相应更新了
        posteriors.push_back(vector<float>(pow(2.0,structSize), 0));
        pCounter.push_back(vector<int>(pow(2.0,structSize), 0));
        nCounter.push_back(vector<int>(pow(2.0,structSize), 0));
    }
}

void FerNNClassifier::getFeatures(const cv::Mat& image,const int& scale_idx, vector<int>& fern){
    int leaf; //叶子  树的最终节点
    //每一个每类器维护一个后验概率的分布，这个分布有2^d个条目（entries），这里d是像素比较pixel comparisons
    //的个数，这里是structSize，即13个comparison，所以会产生2^13即8,192个可能的code，每一个code对应一个后验概率
    for (int t=0;t<nstructs;t++){//nstructs 表示树的个数 10
        leaf=0;
        for (int f=0; f<structSize; f++){
            //struct Feature 特征结构体有一个运算符重载 bool operator ()(const cv::Mat& patch) const
            //返回的patch图像片在(y1,x1)和(y2, x2)点的像素比较值，返回0或者1
            //然后leaf就记录了这13位的二进制代码，作为特征
            leaf = (leaf << 1) + features[scale_idx][t*nstructs+f](image);
        }
        fern[t]=leaf;
    }
}

float FerNNClassifier::measure_forest(vector<int> fern) {
    float votes = 0;
    for (int i = 0; i < nstructs; i++) {
        // 后验概率posteriors[i][idx] =
        //         ((float)(pCounter[i][idx]))/(pCounter[i][idx] + nCounter[i][idx]);
        votes += posteriors[i][fern[i]];//每棵树的每个特征值对应的后验概率累加值 作投票值？？
    }
    return votes;
}

//更新正负样本数，同时更新后验概率
void FerNNClassifier::update(const vector<int>& fern, int C, int N) {
    int idx;
    for (int i = 0; i < nstructs; i++) {
        idx = fern[i];
        (C==1) ? pCounter[i][idx] += N : nCounter[i][idx] += N;
        if (pCounter[i][idx]==0) {
            posteriors[i][idx] = 0;
        } else {
            posteriors[i][idx] = ((float)(pCounter[i][idx]))/(pCounter[i][idx] + nCounter[i][idx]);
        }
    }
}
//随机森林训练
void FerNNClassifier::trainF(const vector<std::pair<vector<int>,int> >& ferns,int resample){
    // Conf = function(2,X,Y,Margin,Bootstrap,Idx)
    //                 0 1 2 3      4         5
    //  double *X     = mxGetPr(prhs[1]); -> ferns[i].first
    //  int numX      = mxGetN(prhs[1]);  -> ferns.size()
    //  double *Y     = mxGetPr(prhs[2]); ->ferns[i].second
    //  double thrP   = *mxGetPr(prhs[3]) * nTREES; ->threshold*nstructs
    //  int bootstrap = (int) *mxGetPr(prhs[4]); ->resample
    ///thr_fern = 0.6
    thrP = thr_fern*nstructs; //6                                      int step = numX / 10;
    //thrN = 0.5*nstructs; //5
    //for (int j = 0; j < resample; j++) {                      // for (int j = 0; j < bootstrap; j++) {
    for (int i = 0; i < ferns.size(); i++){               //   for (int i = 0; i < step; i++) {
        //     for (int k = 0; k < 10; k++) {
        //       int I = k*step + i;//box index
        //       double *x = X+nTREES*I; //tree index
        if(ferns[i].second==1){                           //       if (Y[I] == 1) {
            if(measure_forest(ferns[i].first)<=thrP)      //         if (measure_forest(x) <= thrP)
                update(ferns[i].first,1,1);                 //             update(x,1,1);
        }else{                                            //        }else{
            if (measure_forest(ferns[i].first) >= thrN)   //         if (measure_forest(x) >= thrN)
                update(ferns[i].first,0,1);                 //             update(x,0,1);
        }
    }
    //}
}
void FerNNClassifier::addNN(const vector<cv::Mat>& nn_examples,int pEx_num){
    for(int i = 0; i < pEx_num ;i++)
    {
        pEx.push_back(nn_examples[i]);
    }
    for(int i = pEx_num; i <nn_examples.size() ;i++)
    {
        nEx.push_back(nn_examples[i]);
    }
}

//训练最近邻分类器
void FerNNClassifier::trainNN(const vector<cv::Mat>& nn_examples,int pEx_num){
    float conf,dummy;
     //vector<T> v3(n, i); v3包含n个值为i的元素。y数组元素初始化为0
    vector<int> x(nn_examples.size() - pEx_num,0);
    vector<int> y(pEx_num,1);
    //y[pEx_num - 1]=1;  //上面说到调用trainNN这个函数传入的nn_data样本集，只有一个pEx，在nn_data[0]
    copy(x.begin(),x.end(),back_inserter(y));
    vector<int> isin;
    for (int i=0;i<nn_examples.size();i++){                          //  For each example
         //计算输入图像片与在线模型之间的相关相似度conf
        NNConf(nn_examples[i],isin,conf,dummy);                      //  Measure Relative similarity
        //thr_nn: 0.65 阈值
        //标签是正样本，如果相关相似度小于0.65 ，则认为其不含有前景目标，也就是分类错误了；这时候就把它加到正样本库
        if (y[i]==1 && conf<=thr_nn){                                //    if y(i) == 1 && conf1 <= tld.model.thr_nn % 0.65
            if (isin[1]<0){                                          //      if isnan(isin(2))
                pEx = vector<Mat>(1,nn_examples[i]);                 //        tld.pex = x(:,i);
                continue;                                            //        continue;
            }                                                        //      end
            //pEx.insert(pEx.begin()+isin[1],nn_examples[i]);        //      tld.pex = [tld.pex(:,1:isin(2)) x(:,i) tld.pex(:,isin(2)+1:end)]; % add to model
            pEx.push_back(nn_examples[i]);
        }                                                            //    end
        if(y[i]==0 && conf>0.5)                                      //  if y(i) == 0 && conf1 > 0.5
            nEx.push_back(nn_examples[i]);                             //    tld.nex = [tld.nex x(:,i)];

    }                                                                 //  end
    acum++;
    printf("%d. Trained NN examples: %d positive %d negative\n",acum,(int)pEx.size(),(int)nEx.size());
}                                                                  //  end
///保守相似度csconf
/// 正负样本不相似的最小情况下，负样本不相似所占比例就定义为相关相似度rsconf,csconf < rsconf
///
///  isin中存放三个int型值，初始化全为-1。第一个如果取值为1，则表示NNConf()在计算输入图像片patch
/// 与在线模型pEx中的box时发现在线模型中有一个与其相似度超过阈值ncc_thesame (固定值0.95，
/// 从myParam.yml中读取)，此时会把这个patch也放到在线模型的pEx中，所以第一个取值为1就表示
/// 已经把当前输入图像片patch放到pEx中。第二个的取值依赖于第1个的取值，如果第一个取值为-1，那
/// 么第二个的取值就是-1，如果第一个的取值是1，那么第二个的取值就是在遍历在线模型时找到的第一个
/// 与输入图像片patch相似度超过ncc_the same的box的索引。第三个意义与第一个接近，不同的地方
/// 只在于第一个是对应在线模型的正样本近邻数据集pEx，第三个是对应在线模型的负样本近邻数据集nEx。

void FerNNClassifier::NNConf(const Mat& example, vector<int>& isin,float& rsconf,float& csconf){
    /*Inputs:
   * -NN Patch
   * Outputs:
   * -Relative Similarity (rsconf), Conservative Similarity (csconf), In pos. set|Id pos set|In neg. set (isin)
   */
    isin=vector<int>(3,-1);
    if (pEx.empty()){ //if isempty(tld.pex) % IF positive examples in the model are not defined THEN everything is negative
        rsconf = 0; //    conf1 = zeros(1,size(x,2));
        csconf=0;
        return;
    }
    if (nEx.empty()){ //if isempty(tld.nex) % IF negative examples in the model are not defined THEN everything is positive
        rsconf = 1;   //    conf1 = ones(1,size(x,2));
        csconf=1;
        return;
    }
    Mat ncc(1,1,CV_32F);
    float nccP,csmaxP,maxP=0;
    bool anyP=false;
      //ceil返回大于或者等于指定表达式的最小整数
    int maxPidx,validatedPart = ceil(pEx.size()*valid);
    float nccN, maxN=0;
    bool anyN=false;
    //比较图像片p到在线模型M的距离（相似度），计算正样本最近邻相似度，也就是将输入的图像片与
    //在线模型中所有的图像片进行匹配，找出最相似的那个图像片，也就是相似度的最大值
    for (int i=0;i<pEx.size();i++){

        matchTemplate(pEx[i],example,ncc,CV_TM_CCORR_NORMED);      // measure NCC to positive examples
        nccP=(((float*)ncc.data)[0]+1)*0.5; //计算匹配相似度
        if (nccP>ncc_thesame)
            anyP=true;
        if(nccP > maxP){
            maxP=nccP;//记录最大的相似度以及对应的图像片index索引值
            maxPidx = i;
            if(i<validatedPart)
                csmaxP=maxP;
        }
    }
      //计算负样本最近邻相似度
    for (int i=0;i<nEx.size();i++){
        matchTemplate(nEx[i],example,ncc,CV_TM_CCORR_NORMED);     //measure NCC to negative examples
        nccN=(((float*)ncc.data)[0]+1)*0.5;
        if (nccN>ncc_thesame)
            anyN=true;
        if(nccN > maxN)
            maxN=nccN;
    }
    //set isin
    if (anyP) isin[0]=1;  //if he query patch is highly correlated with any positive patch in the model then it is considered to be one of them
    isin[1]=maxPidx;      //get the index of the maximall correlated positive patch
    if (anyN) isin[2]=1;  //if  the query patch is highly correlated with any negative patch in the model then it is considered to be one of them
    //Measure Relative Similarity
     //相关相似度 = 正样本最近邻相似度 / （正样本最近邻相似度 + 负样本最近邻相似度）
    float dN=1-maxN;
    float dP=1-maxP;
    rsconf = (float)dN/(dN+dP);
    //Measure Conservative Similarity
    dP = 1 - csmaxP;
    csconf =(float)dN / (dN + dP);
}

void FerNNClassifier::evaluateTh(const vector<pair<vector<int>,int> >& nXT,const vector<cv::Mat>& nExT){
    float fconf;
    for (int i=0;i<nXT.size();i++){
        fconf = (float) measure_forest(nXT[i].first)/nstructs;
        if (fconf>thr_fern)
            thr_fern=fconf;
    }
    vector <int> isin;
    float conf,dummy;
    for (int i=0;i<nExT.size();i++){
        NNConf(nExT[i],isin,conf,dummy);
        if (conf>thr_nn)
            thr_nn=conf;

    }
    if (thr_nn>thr_nn_valid)
        thr_nn_valid = thr_nn;
}

void FerNNClassifier::show(){
    Mat examples((int)pEx.size()*pEx[0].rows,pEx[0].cols,CV_8U);
    double minval;
    Mat ex(pEx[0].rows,pEx[0].cols,pEx[0].type());
    for (int i=0;i<pEx.size();i++)
    {
        minMaxLoc(pEx[i],&minval);
        pEx[i].copyTo(ex);

        ex = ex-minval;

        Mat tmp = examples.rowRange(Range(i*pEx[i].rows,(i+1)*pEx[i].rows));
        ex.convertTo(tmp,CV_8U);
    }
    imshow("Examples",examples);
}
void FerNNClassifier::storeNP(bool isStart)
{
    char ExName[1024];
    int i = 0;

    FileStorage fs("D:\\opencv_project\\OpenTLD\\data\\PN_data.yml", FileStorage::WRITE);

    for (i = 0; i < pEx.size(); i++)
    {
        //sprintf(ExName,"D:\\opencv_project\\OpenTLD\\data\\P\\%d.jpg",i);
        //imwrite(ExName,pEx[i]);
        sprintf(ExName,"pEx%d",i);
        fs << ExName << pEx[i];
    }
    fs << "pExtotal" << i;
    for (i = 0; i < nEx.size(); i++)
    {
        sprintf(ExName,"nEx%d",i);
        fs << ExName << nEx[i];
        //sprintf(ExName,"D:\\opencv_project\\OpenTLD\\data\\N\\%d.jpg",i);
        //imwrite(ExName,nEx[i]);
    }
    fs << "nExtotal" << i;
    fs.release();
}
