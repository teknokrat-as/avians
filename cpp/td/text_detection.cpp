#include "text_detection.hpp"

namespace avians
{

    Mat load_image(string img_f)
    {
        return cv::imread(img_f, 0);
    }

    AviansTD::AviansTD(string model_fname1, 
                       string model_fname2) {
        this->model_fname1 = model_fname1;
        this->model_fname2 = model_fname2;
        load_filters();
    }

    void AviansTD::load_filters() {
        this->filter1 = createERFilterNM1(loadClassifierNM1(this->model_fname1),
                                          8,
                                          0.00015f,
                                          0.13f,
                                          0.2f,
                                          true,
                                          0.1f);
        this->filter2 = createERFilterNM2(loadClassifierNM2(this->model_fname2),
                                          0.5);
        
    }

    
    vector<vector<ERStat> > AviansTD::find_text_regions(string image_fname) {
        Mat src = cv::imread(image_fname);
        vector<Mat> channels;
        computeNMChannels(src, channels);
        
        int n_channels = (int) channels.size();
        for (int c = 0; c < n_channels - 1; c++)
            channels.push_back(255 - channels[c]);
        
        n_channels = (int) channels.size();
        vector<vector<ERStat> > regions(n_channels);
        
        for (int c=0; c < n_channels; c++)
        {
            er_filter1->run(channels[c], regions[c]);
            er_filter2->run(channels[c], regions[c]);
        }

        return regions;
        

    }
        
    void AviansTD::write_regions(vector<Mat> regions, const string output_dir) {
    
    }
}
