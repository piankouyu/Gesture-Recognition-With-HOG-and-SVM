#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <fstream>
#include <filesystem>
using namespace std;
using namespace cv;
class clPreProcessing {
public:
    clPreProcessing(Mat img=Mat()) {
        this->img = img;
    }
    Mat processHSV(Mat img, bool rmask = false, vector<int> val = { 0,100,0,20,255,255,135,31,18,180,255,255 }) {
        cvtColor(img,this->img, COLOR_BGR2HSV);
        Scalar lower_skin(val[0], val[1], val[2]);
        Scalar upper_skin(val[3], val[4], val[5]);
        Mat mask1;
        inRange(this->img, lower_skin, upper_skin, mask1);
        lower_skin = Scalar(val[6], val[7], val[8]);
        upper_skin = Scalar(val[9], val[10], val[11]);
        Mat mask2;
        inRange(this->img, lower_skin, upper_skin, mask2);
        Mat mask;
        bitwise_or(mask1, mask2, mask);
        Mat kernel = cv::Mat::ones(3, 3, CV_8U);
        Mat kernel1 = cv::Mat::ones(5, 5, CV_8U);
        morphologyEx(mask,mask, MORPH_OPEN, kernel);
        morphologyEx(mask,mask, MORPH_DILATE, kernel1);
        // apply mask on the original image
        bitwise_and(img, img,img, mask);
        if (rmask) {
            this->img = mask;
        }
        return this->img;
    }

    // process filter image
    Mat processFilter(Mat img) {
        Mat kernel = Mat::ones(5, 5, CV_8U);
        // smooth the image
        medianBlur(img,img, 7);
        dilate(this->img, this->img, kernel, Point(-1, -1), 5);
        morphologyEx(this->img, this->img, MORPH_ELLIPSE, kernel);
        return this->img;
    }

    // process YCrBr image
    Mat processYCrBr(Mat img, bool rmask = false, vector<int> val = { 27 ,133 ,28 ,135 ,167 ,145 ,13 ,38 ,20 ,37 ,22 ,12 }) {
        cvtColor(img, this->img, COLOR_BGR2YCrCb);
        Scalar lower_skin(val[0], val[1], val[2]);
        Scalar upper_skin(val[3], val[4], val[5]);
        Mat mask1;
        inRange(this->img, lower_skin, upper_skin, mask1);
        lower_skin = Scalar(val[6], val[7], val[8]);
        upper_skin = Scalar(val[9], val[10], val[11]);
        Mat mask2;
        inRange(this->img, lower_skin, upper_skin, mask2);
        Mat mask;
        bitwise_or(mask1, mask2, mask);
        Mat kernel = cv::Mat::ones(3, 3, CV_8U);
        Mat kernel1 = cv::Mat::ones(5, 5, CV_8U);
        morphologyEx(mask,mask, MORPH_OPEN, kernel);
        morphologyEx(mask,mask, MORPH_DILATE, kernel1);
        bitwise_and(this->img, this->img, this->img, mask);
        if (rmask) {
            this->img = mask;
        }
        return this->img;
    }

    // combine detections
    Mat CombineDetections(Mat img) {
        Mat hsvm = this->processHSV(img, true);
        Mat ycrbrm = this->processYCrBr(img, true);
        Mat masks;
        bitwise_or(hsvm, ycrbrm, masks);
        medianBlur(masks,masks,23);
        bitwise_and(img, img,this->img, masks);
        return this->img;
    }
private:
    Mat img; // image matrix
};
class clTraningSetManager {
public:
    clTraningSetManager() {}
    // read training directory
    vector<string> ReadTrainingDirectory(string dir) {
        vector<string> pathlist;
        vector<string> labellist;
        int count = 0;
        vector<string> labelsAndIDs;
        // get all directories
        for (const auto& entry : filesystem::directory_iterator(dir)) {
            if (entry.is_directory()) {
                pathlist.push_back(entry.path().string());
                labellist.push_back(entry.path().filename().string());
            }
        }
        // create list
        for (const auto& label : labellist) {
            count++;
            string id = to_string(count);
            string dir = pathlist[count - 1];
            labelsAndIDs.push_back(id + "," + label + "," + dir);
        }
        return labelsAndIDs;
    }
    // save labels file
    void SaveLabelsFile(string dir, string file) {
        vector<string> lai = this->ReadTrainingDirectory(dir);
        // save data into the file
        ofstream outfile(file);
        for (const auto& i : lai) {
            outfile << i << "\n";
        }
        outfile.close();
    }
    // save calibration
    void SaveCalibration(string labelfile, vector<int> calval = {}) {
        // read all file
        ifstream infile(labelfile);
        vector<string> lines;
        string line;
        while (getline(infile, line)) {
            lines.push_back(line);
        }
        infile.close();
        // write lines without calibration values
        ofstream outfile(labelfile);

        for (const auto& line : lines) {
            if (line.find("#cal,") == string::npos) {
                outfile << line << "\n";
            }
        }
        outfile.close();
        // add to the end the values
        outfile.open(labelfile, ios_base::app);
        int h = calval[0];
        int s = calval[1];
        int v = calval[2];
        string ws = to_string(h) + "," + to_string(s) + "," + to_string(v);
        outfile << "#cal," << ws << "\n";
        outfile.close();
    }

    // load labels file
    vector<string> LoadLabelsFile(string file, bool calibration = false) {
        vector<string> lf;
        ifstream infile(file);
        string line;

        while (getline(infile, line)) {
            if (calibration == false) {
                // split data, skip what if needed
                if (line.find('#') == string::npos) {
                    stringstream ss(line);
                    vector<string> res;
                    while (ss.good()) {
                        string substr;
                        getline(ss, substr, ',');
                        res.push_back(substr);
                    }
                    lf.insert(lf.end(), res.begin(), res.end());
                }
            }
            else {
                if (line.find("#cal, ") != string::npos) {
                    // get calibration values
                    line.erase(0, 5); // remove #cal,
                    stringstream ss(line);
                    string h, s, v;
                    ss >> h >> s >> v; // parse values
                    lf = { h,s,v };
                }
            }
        }
        infile.close();
        return lf;
    }
};
class ContourDetector {
public:
    // constructor
    ContourDetector() {
        gimg = Mat();
        font = FONT_HERSHEY_SIMPLEX;
        items = vector<vector<int>>();
        count = 0;
    }

    // contour filter
    vector<vector<int>> ContourFilter(Mat img, double area = 1000.0) {

        this->items.clear();
        cvtColor(img, this->gimg, COLOR_BGR2GRAY);

        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;
        findContours(this->gimg, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

        int id = 1;
        for (const auto& cnt : contours) {
            double larea = cv::contourArea(cnt);
            if (larea > area) {
                int x, y, w, h;
                cv::Rect rect = boundingRect(cnt);
                x = rect.x;
                y = rect.y;
                w = rect.width;
                h = rect.height;
                cout << "ROI" << x << ' ' << y << ' ' << w << ' ' << h << endl;
                cv::Point center(x + w / 2, y + h / 2);
                this->items.push_back({ id, x, y, w, h, center.x, center.y });
                id++;
            }
        }
        return this->items;
    }

    // draw detections
    Mat DrawDetections(Mat img, vector<vector<int>> detections, int offset = 20, bool objCenter = true, bool objRectangle = true, vector<vector<int>> label = { {0,0} }, bool drawline = false) {
        // copy image
        Mat limg;
        img.copyTo(limg);
        vector<Point> arr;
        // check if there are detections
        if (!detections.empty()) {
            for (const auto& ii : detections) {
                int id, x, y, w, h, cx, cy;
                id = ii[0];
                x = ii[1];
                y = ii[2];
                w = ii[3];
                h = ii[4];
                cx = ii[5];
                cy = ii[6];
                Point center(cx, cy);
                if (objCenter) {
                    circle(limg, center, 2, Scalar(0, 255, 0), -1);
                }
                if (objRectangle) {
                    rectangle(limg, Point(x - offset, y - offset), Point(x + w + offset, y + h + offset), Scalar(0, 255, 0), 1);
                }
                for (const auto& ll : label) {
                    if (ll[0] == id) {
                        putText(limg, to_string(ll[1]), Point(x - offset + 5, y - offset + 15), this->font, 0.4, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
                    }
                }
                arr.push_back(center);
            }

            if (arr.size() > 2 && drawline == true) {
                drawContours(limg, arr, 0, Scalar(255, 0, 255), 3);
            }

            return limg;
        }
    }

    // get ROI for detections
    vector<Mat> GetRoiForDetections(Mat img, vector<vector<int>> detections, int offset = 20, Size roi_size = Size(96, 96)) {

        vector<cv::Mat> rois;
        // check if there are detections
        if (!detections.empty()) {
            for (const auto& ii : detections) {
                int id, x, y, w, h, cx, cy;
                id = ii[0];
                x = ii[1];
                y = ii[2];
                w = ii[3];
                h = ii[4];
                cx = ii[5];
                cy = ii[6];

                try {
                    Mat detRoi = img(Rect(x - offset, y - offset, w + 2 * offset, h + 2 * offset));
                    normalize(detRoi, detRoi, 0, 1, NORM_MINMAX, CV_32F);
                    resize(detRoi, detRoi, roi_size);
                    rois.push_back(detRoi);
                }
                catch (...) {
                    // do nothing
                }
            }
            return rois;
        }
    }

    // show ROIs on image
    void ShowRoisOnImage(Mat img, vector<Mat> rois, Size roi_size = Size(96, 96)) {

        if (!rois.empty()) {
            int offs = 0;
            try {
                for (const auto& ii : rois) {
                    normalize_restore(ii, ii);
                    img(Rect(offs, 0, roi_size.width, roi_size.height)) = ii;
                    offs += roi_size.width;
                }
            }
            catch (...) {
                // do nothing
            }
        }
    }

    // save images
    void SaveImages(Mat img, vector<Mat> rois, string path = ".", int initnumber = 0, bool usetime = true, bool saveframe = false, string prefix = "") {
        time_t t = time(nullptr);
        struct tm* now = localtime(&t);
        stringstream timeStr;
        timeStr << now->tm_hour;
        timeStr << now->tm_min;
        timeStr << now->tm_sec;
        string time = timeStr.str();

        if (!rois.empty()) {
            for (const auto& ii : rois) {
                normalize_restore(ii, ii);
                if (usetime == true) {
                imwrite(path + "img_" + prefix + time + ".png", ii);
                }
                else {
                imwrite(path + "img_" + prefix + to_string(this->count + initnumber) + ".png", ii);
                    this->count++;
                }
            }
        }
        else {
        imwrite(path + "img_" + prefix + time + ".png", img);
        }
    }

    void normalize_restore(Mat src, Mat dst, double max=255, double min=128)
    { // 检查输入图像是否为空
        if (src.empty())
        {
            cout << "Input image is empty." << endl;
            return;
        } // 检查输入图像是否为浮点类型
        if (src.type() != CV_32F && src.type() != CV_64F)
        {
            cout << "Input image must be float type." << endl;
            return;
        }                                                  // 创建输出图像，与输入图像大小和通道数相同，但类型为无符号字符
        dst = Mat(src.size(), CV_8UC(src.channels())); // 遍历输入图像的每个像素
        for (int i = 0; i < src.rows; i++)
        {
            for (int j = 0; j < src.cols; j++)
            { // 对于每个通道，用反向公式计算还原后的像素值，并赋值给输出图像
                for (int k = 0; k < src.channels(); k++)
                {
                    dst.at<Vec3b>(i, j)[k] = static_cast<uchar>(src.at<Vec3f>(i, j)[k] * (max - min) + min);
                }
            }
        }
    }
private:
    // data members
    Mat gimg; // grayscale image matrix
    int font; // font type
    vector<vector<int>> items; // contour items
    int count; // image count
};
class clAutoCalibrate {
public:
    clAutoCalibrate() {
        img = Mat();
        h = 0;
        s = 0;
        v = 0;
        font = FONT_HERSHEY_SIMPLEX;
    }
    tuple<int, int, int> ProvideClaibParams() {
        return make_tuple(h, s, v);
    }
    Mat RunCalibration(Mat img) {
        int x = 10, y = 10;
        int w = 80, h = 80;
        Mat hsv;
        cvtColor(img, hsv, cv::COLOR_BGR2HSV);
        rectangle(img, cv::Point(8, 8), Point(90, 90), Scalar(0, 255, 0), 2);
        Mat detRoi = hsv(cv::Rect(x, y, w, h));
        vector<Mat> channels;
        split(detRoi, channels);
        h = static_cast<int>(cv::mean(channels[0])[0]);
        s = static_cast<int>(cv::mean(channels[1])[0]);
        v = static_cast<int>(cv::mean(channels[2])[0]);
        detRoi.copyTo(img(Rect(x, y, w, h)));
        string msg = "hsv=" + to_string(h) + "," + to_string(s) + "," + to_string(v);
        putText(img, msg, Point(5, 105), font, 0.4, cv::Scalar(0, 255, 0), 1, LINE_AA);
        return img;
    }
private:
    Mat img;
    int h;
    int s;
    int v;
    int font = cv::FONT_HERSHEY_SIMPLEX;
};
class clHogDetector {
public:
    // constructor
    clHogDetector(int sampleSize = 64, std::string fn = "") {
        Size winSize(sampleSize, sampleSize);
        Size blockSize(16, 16);
        Size blockStride(8, 8);
        Size cellSize(8, 8);
        int nbins = 9;
        int derivAperture = 1;
        double winSigma = 4.;
        int histogramNormType = 0;
        double L2HysThreshold = 2.0000000000000001e-01;
        bool gammaCorrection = false;
        int nlevels = 64;

        this->hog = HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
            HOGDescriptor::L2Hys, L2HysThreshold, gammaCorrection);
        this->templates = vector<tuple<int, string, string>>();
        this->HogAccumulator = vector<tuple<int, Mat, string>>();
        this->LabelList = vector<tuple<int, string>>();
        this->svm = cv::ml::SVM::create();
        // create SVM
        if (fn == "") {
            // do nothing
        }
        else {
            this->svm = cv::ml::SVM::load(fn);
        }

        // n-class classification
        this->svm->setType(cv::ml::SVM::C_SVC);

        // Binary classification (detections belong to one or other class)
        this->svm->setKernel(cv::ml::SVM::LINEAR);

        // termination criteria
        this->svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 100, 1e-6));
    }

    // add label and files
    void AddLabelAndFiles(std::string FileName, int label, std::string strlabel) {

        this->templates.push_back(std::make_tuple(label, FileName, strlabel));
    }

    // get HOG for an image
    Mat GetHogForAnImage(cv::Mat img, int sampleSize = 64) {

        Mat resized;
        resize(img, resized, cv::Size(sampleSize, sampleSize), 0, 0, cv::INTER_AREA);
        imshow("HOG", resized);
        waitKey(0);
        vector<float> ret;
        this->hog.compute(resized, ret);

        return fvector2fmat(ret);
    }

    // img float to int
    cv::Mat ImgFloatToInt(cv::Mat img) {

        cv::Mat intimg;
        img.convertTo(intimg, CV_8U, 255, 0);

        return intimg;
    }

    // get images HOG features
    void GetImagesHOGFeatures(int sampleSize = 64) {

        for (const auto& i : this->templates) {
            // get files from disk
            int label = get<0>(i);
            string FileName = get<1>(i);
            string strlabel = get<2>(i);

            Mat img = cv::imread(FileName, cv::IMREAD_GRAYSCALE);

            normalize(img, img,  0, 1, NORM_MINMAX,CV_32F);

            img = this->ImgFloatToInt(img);

            // resize it
            Mat resized;
            resize(img, resized, Size(sampleSize, sampleSize), 0, 0, INTER_AREA);
            //imshow("HOG", resized);
            //waitKey(0);
            vector<float> ret;
            this->hog.compute(resized, ret);
            cout << ret.size() << endl;
            this->HogAccumulator.push_back(make_tuple(label, fvector2fmat(ret), strlabel));
        }
    }
    Mat fvector2fmat(vector<float> output) {
        Mat out_result(1, output.size(), CV_32FC1, cv::Scalar(0));
        memcpy(out_result.data, output.data(), output.size() * sizeof(float));
        return out_result;
    }
    // get HOG accumulator
    vector<tuple<int, Mat, string>> GetHogAccumulator() {

        return this->HogAccumulator;
    }

    // update label names
    void UpdateLabelNames(vector<string> labels) {

        for (int i = 0; i < labels.size(); i += 3) {
            int id = atoi(labels[i].c_str());
            string l = labels[i + 1];
            string dir = labels[i + 2];
            this->LabelList.push_back(std::make_tuple(id, l));
        }
    }

    // train SVM with HOG
    void TrainSVMWithHOG(int sampleSize = 64) {


        // compute training set hog features and add to accumulator
        this->GetImagesHOGFeatures(sampleSize = sampleSize);

        Mat trainingData;
        Mat trainingCalss;

        for (const auto& i : this->HogAccumulator) {
            int pn = get<0>(i);
            Mat hogv = get<1>(i);
            string strlabel = get<2>(i);
            hogv = hogv.reshape(1,1); // flatten the matrix

            trainingData.push_back(hogv); // append row
            trainingCalss.push_back(pn); // append class label
        }

        trainingData.convertTo(trainingData, CV_32F); // convert to float
        trainingCalss.convertTo(trainingCalss, CV_32S); // convert to int
        printf("Data row:%d col:%d Class row:%d\r\n", trainingData.rows, trainingData.cols, trainingCalss.rows);

        this->svm->train(trainingData,  ml::ROW_SAMPLE,trainingCalss);

    }

    // read training files
    vector<string> ReadTrainingFiles(string dir) {
    vector<string> paths;
    for (const auto& entry : filesystem :: directory_iterator(dir)) {
        if (entry.is_regular_file()) {
            paths.push_back(entry.path().string());
            //print(path)
        }
    }
    return paths;
}

// add to training set
void AddToTrainingSet(string dir, int label, string strlabel = "") {


    vector<string> f = this->ReadTrainingFiles(dir);
    for (const auto& a : f) {
        this->AddLabelAndFiles(a, label, strlabel);
    }
}

// classify ROI
vector<tuple<int, string>> ClassifyRoi(vector<Mat> imgs = Mat(), int sampleSize = 64) {
    vector<tuple<int, string>> arrFound;
    int id = 0;
    for (const auto& r : imgs) {
    // histogram normalization
        Mat bw,ival;
        cvtColor(r,bw, COLOR_BGR2GRAY);
        bw = this->ImgFloatToInt(bw);
        equalizeHist(bw,ival);
        Mat ihog = this->GetHogForAnImage(ival, sampleSize = sampleSize);
        ihog = ihog.reshape(1,1); // flatten the matrix

        id++;
        int val = this->svm->predict(ihog);
        string valstr;
        valstr = this->GetStrLabelByClassificationID(val);
        arrFound.push_back(make_tuple(id, valstr));
    }   
return arrFound;
}

// get string label by classification ID
string GetStrLabelByClassificationID(int val) {

    string ret = to_string(val);
    for (const auto& i : this->LabelList) {
        int v = get<0>(i);
        string l = get<1>(i);
        if (v == val) {
            if (l == "") {
                ret = v;
            }
            else {
                ret = l;
            }
            break;
        }
    }   
return ret;
}


// save training data
void SaveTrainingData(string fn = "./data.xml") {
    this->svm->save(fn);
}

private:
    // data members
    HOGDescriptor hog; // HOG descriptor object
    vector<tuple<int, string, string>> templates; // template images with labels and file names
    vector<tuple<int, Mat, string>> HogAccumulator; // HOG features with labels and file names
    vector<tuple<int, string>> LabelList; // label names
    Ptr<ml::SVM> svm; // SVM object
};

int main()
{
    ////train
    // 
    string traindir = "./data/train";
    string labelfile = "./labels.txt";
    string trainedfile = "./data.xml";
    clTraningSetManager TSM;
    TSM.SaveLabelsFile(traindir, labelfile);
    cout << "Labelfile " << labelfile << " created, exiting.";
    clHogDetector HD(96);
    vector<string> lf = TSM.LoadLabelsFile("./labels.txt");
    for (int i = 0; i < lf.size(); i += 3) {
        HD.AddToTrainingSet(lf[i + 2], atoi(lf[i].c_str()), lf[i + 1]);
    }
    HD.UpdateLabelNames(lf);
    HD.TrainSVMWithHOG(96);
    HD.SaveTrainingData(trainedfile);

    //predict
    // 
    //clPreProcessing PP;
    //ContourDetector CD;
    //clTraningSetManager TSM;
    //clHogDetector HD(96, "./data.xml");
    //vector<string> lf = TSM.LoadLabelsFile("./labels.txt");
    //HD.UpdateLabelNames(lf);

    //for (auto& p : filesystem::directory_iterator("./data/test/V")) {
    //    Mat img0 = imread(p.path().string());
    //    //Mat img1 = PP.CombineDetections(img0);
    //    //img1 = PP.processFilter(img1);
    //    auto detection = CD.ContourFilter(img0, 500);
    //    auto rois = CD.GetRoiForDetections(img0, detection, 0);
    //    
    //    auto res = HD.ClassifyRoi(rois, 96);
    //    for (auto& i : res) {
    //        cout << get<0>(i) << " " << get<1>(i) << endl;
    //    }
    //}
}