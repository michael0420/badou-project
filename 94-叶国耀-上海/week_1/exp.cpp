// opencv_demo.cpp
#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;
int main()
{
// cout << "OpenCV Version: " << CV_VERSION << endl;
Mat img = imread("lenna.png");
// typedef BOOST_TYPEOF(*img.data) ElementType;
int rows = img.rows;
int cols = img.cols;
int channels = img.channels();
cout << rows << ' '<< cols << ' ' << channels << endl;

Mat img_grey = Mat_<uchar>(rows, cols);
for (int row=0; row<rows; row++)
{
    for (int col=0; col<cols; col++)
    {
        img_grey.at<uchar>(row,col) = uchar((img.at<Vec3b>(row,col)[0]*0.11+
                                        img.at<Vec3b>(row,col)[1]*0.59+
                                        img.at<Vec3b>(row,col)[2]*0.3));
    }
}

Mat img_binary = Mat_<uchar>(rows, cols);
for (int row=0; row<rows; row++)
{
    for (int col=0; col<cols; col++)
    {
        if (int(img_grey.at<uchar>(row,col)) >= 128)
        {
            img_binary.at<uchar>(row,col) = uchar(255);
        }
        else
        {
            img_binary.at<uchar>(row,col) = uchar(0);
        }
    }
}
// cvtColor(img, img_grey, CV_BGR2GRAY);
cout << rows << ' '<< cols << ' ' << img_grey.channels() << endl;
cout << int(img_grey.at<uchar>(0,0)) << ' '<< int(img_grey.at<uchar>(126,126)) << ' ' << int(img_grey.at<uchar>(500,500)) << endl;
imshow("BGR", img);
imshow("Grey", img_grey);
imshow("Binary", img_binary);
waitKey(0);
return 0;
}
