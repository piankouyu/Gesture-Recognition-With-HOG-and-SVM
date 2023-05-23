# Gesture-Recognition-With-HOG-and-SVM
## About
基于C++ 17和Opencv 4.5.4的传统手势识别

The project presents a simple solution for retraining and classifying images with SVM and HOG features. Currently the tranined file can detect and classify multiple hand gestures.
## Retrain
#### 1. Create a directory structure
Is recommended to start from the root of the cpp file, and create a directory like "data", then create under these folder directories equivalent to the classes that are needed for the classification part. Similar to below.
```
├── data
│   ├── A
│   ├── C
│   ├── Five
│   ├── V
```
#### 2. Run code to retrain
The project provides functions to generate label file, then the rest is easy.
```
int main(){
    //train
    string traindir = "./data";
    string labelfile = "./labels.txt";
    string trainedfile = "./data.xml";
    clTraningSetManager TSM;
    TSM.SaveLabelsFile(traindir, labelfile);
    cout << "Labelfile " << labelfile << " created, exiting.";
    clHogDetector HD(96);
    vector<string> lf = TSM.LoadLabelsFile("./labels.txt");
    for (int i = 0; i < lf.size(); i += 3) {
        HD.AddToTrainingSet(lf[i], atoi(lf[i + 1].c_str()), lf[i + 2]);
    }
    HD.UpdateLabelNames(lf);
    HD.TrainSVMWithHOG(96);
    HD.SaveTrainingData(trainedfile);
}
```
## Resources
https://github.com/fvilmos/gesture_detector

/Thanks
