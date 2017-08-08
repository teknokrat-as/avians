
#define CATCH_CONFIG_MAIN

#include "text_detection.hpp"
#include "../common/catch.hpp"

#include <cstdio>
using namespace avians;


const string TEST_IMAGE_FILE = "/home/iesahin/Annex/Arabic/working-set-1/rt-1-img8094.jpg";;
const string OUTPUT_DIR = "/dev/shm/test_detection_output/";
const string NMFILE1 = "/home/iesahin/Annex/Arabic/td_trained_arbc.xml";
const string NMFILE2 = "/home/iesahin/Annex/Arabic/td_trained_arbc2.xml";


TEST_CASE("Image is loaded", "[image]")
{
    Mat img = load_image(TEST_IMAGE_FILE);
    REQUIRE(img.cols > 0);
    REQUIRE(img.rows > 0);
}


SCENARIO("Text is detected on the image", "[td]" ) {
    vector<Mat> regions;

    GIVEN("Model files are supplied as parameter") {
            AviansTD avtd = AviansTD(NMFILE1, NMFILE2);
            WHEN("The program starts") {
                THEN("Models are loaded") {
                    REQUIRE(avtd.filter1 != 0);
                    REQUIRE(avtd.filter2 != 0); 
                }
            }

            WHEN("An image is supplied to find_regions") {
                regions = avtd.find_text_regions(TEST_IMAGE_FILE);
                THEN("Image is loaded") {
                    REQUIRE(avtd.last_image().rows > 0);
                    REQUIRE(avtd.last_image().cols > 0); }
                THEN("Regions are returned") {
                    REQUIRE(regions.size() > 0); 
                }
            }

            WHEN("A folder supplied to store resulting images") {
                avtd.write_regions(regions, OUTPUT_DIR);
                THEN("Number of files in OUTPUT_DIR is equal to number of regions") {
                    path output_path(OUTPUT_DIR);
                    int n_files = 0;
                    for (directory_iterator itr(output_path); 
                         itr!=directory_iterator(); ++itr) {
                        if (is_regular_file(itr->status())) n_files++; 
                    }

                    REQUIRE(n_files == regions.size());
                }
            }

        }
}                    
