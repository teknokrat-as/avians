
#include <opencv2/text.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <boost/filesystem.hpp>

#include <iostream>

using namespace std;
using namespace cv;
using namespace cv::text;
using namespace boost::filesystem;

namespace avians

{

    class AviansTD {

    public: 
        Ptr<ERFilter> filter1, filter2;error: ambiguous overload for ‘operator<<’error:
        string model_fname1, model_fname2;
    
        AviansTD(string model_fname1, 
                 string model_fname2);    

        void load_filters();
        vector<Mat> find_text_regions(string image_fname);
        Mat last_image();
        void write_regions(vector<Mat> regions, const string output_dir);
    };

    Mat load_image(string img_f);
}
