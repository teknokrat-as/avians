
cmake_minimum_required( VERSION 2.8 )

project( avians-td )

set(${PROJECT_NAME}_MAJOR_VERSION 1})
set(${PROJECT_NAME}_MINOR_VERSION 0})

#set(CMAKE_BUILD_TYPE Release)
set(CMAKE_BUILD_TYPE Debug)

find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

# find_package(PythonLibs REQUIRED)
# include_directories (${PYTHON_INCLUDE_DIRS})

find_package(Boost 1.58.0 COMPONENTS python REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

# add_library(pytest SHARED pytest.cpp)
# target_link_libraries(pytest
#   boost_python
#   ${Boost_LIBRARIES} 
#   ${PYTHON_LIBRARIES})

add_executable(avians-td
  text_detection)

target_link_libraries(avians-td
  ${OpenCV_LIBS}
  ${Boost_LIBRARIES})

add_executable(test_text_detection
  test_text_detection
  text_detection)

target_link_libraries(test_text_detection
  ${OpenCV_LIBS}
  ${Boost_LIBRARIES})

add_custom_command(TARGET test_text_detection
  POST_BUILD
  COMMAND test_text_detection)


# set(DEPENDENCY_DIR "${PROJECT_SOURCE_DIR}/../dependencies")
# set(OPENCV_DIR "${DEPENDENCY_DIR}/OpenCV/")

# message(STATUS)
# message(STATUS "DEPENDENCY_DIR:" ${DEPENDENCY_DIR})
# message(STATUS)

# include_directories(${DEPENDENCY_DIR}/boost/)
# include_directories(${OPENCV_DIR}/include)
# include_directories(${OPENCV_DIR}/modules/core/include)
# include_directories(${OPENCV_DIR}/modules/calib3d/include)
# include_directories(${OPENCV_DIR}/modules/contrib/include)
# include_directories(${OPENCV_DIR}/modules/features2d/include)
# include_directories(${OPENCV_DIR}/modules/flann/include)
# include_directories(${OPENCV_DIR}/modules/gpu/include)
# include_directories(${OPENCV_DIR}/modules/highgui/include)
# include_directories(${OPENCV_DIR}/modules/imgproc/include)
# include_directories(${OPENCV_DIR}/modules/legacy/include)
# include_directories(${OPENCV_DIR}/modules/ml/include)
# include_directories(${OPENCV_DIR}/modules/nonfree/include)
# include_directories(${OPENCV_DIR}/modules/objdetect/include)
# include_directories(${OPENCV_DIR}/modules/ocl/include)
# include_directories(${OPENCV_DIR}/modules/photo/include)
# include_directories(${OPENCV_DIR}/modules/video/include)

# link_directories(${DEPENDENCY_DIR}/boost/stage/lib)
# link_directories(${OPENCV_DIR}/lib)

# if (MSVC)
# 	set(BSUFFIX -vc110-mt-gd-1_53)
# 	set(OTAP_BOOST_LIBS libboost_system${BSUFFIX} libboost_serialization${BSUFFIX} libboost_filesystem${BSUFFIX} libboost_program_options${BSUFFIX} libboost_random${BSUFFIX} libboost_thread${BSUFFIX} libboost_regex${BSUFFIX} ) 
# 	set(OSUFFIX 244d)
# 	set(OTAP_OPENCV_LIBS opencv_core${OSUFFIX} opencv_features2d${OSUFFIX} opencv_highgui${OSUFFIX} opencv_imgproc${OSUFFIX} opencv_nonfree${OSUFFIX})
# 	set(OTAP_COMMON_LIBS ${OTAP_BOOST_LIBS} ${OTAP_OPENCV_LIBS} ${OTAP_MODULE_LIBS})
# 	set(OTAP_TESTING_LIBS ${OTAP_COMMON_LIBS} libboost_unit_test_framework)
	
# else()
# 	set(OTAP_BOOST_LIBS boost_system boost_serialization boost_filesystem boost_program_options boost_random boost_thread boost_regex)  
# 	set(OTAP_OPENCV_LIBS opencv_core opencv_features2d opencv_highgui opencv_imgproc opencv_nonfree)
# 	set(OTAP_COMMON_LIBS ${OTAP_BOOST_LIBS} ${OTAP_OPENCV_LIBS} ${OTAP_MODULE_LIBS})
# 	set(OTAP_TESTING_LIBS ${OTAP_COMMON_LIBS} boost_unit_test_framework)
# endif()    


# # set(OTAP_MODULE_LIBS otap-database)


# ## USE GCC Opts

# if (CMAKE_COMPILER_IS_GNUCXX)
# 	add_definitions(-std=c++11)
# 	add_definitions(-Wall)
# endif()


# # add_definitions(-Wpedantic)
  
# # add_subdirectory(otap-binarize)

# add_subdirectory(otap-word-spot)

# # add_subdirectory(otap-database)


# # include_directories(${PROJECT_SOURCE_DIR}/otap-database)

# set(DERVAZE_COMMON_CPP_VERSION_MAJOR 1)
# set(DERVAZE_COMMON_CPP_VERSION_MINOR 1)

# add_executable(cppdeneme cppdeneme)

## Following are from OTAP, kept as examples to write. 

# set(CC_FILES connected-component connected-component-library line-pair-feature single-line-feature shape-context-feature sift-feature surf-feature freak-feature orb-feature chamfer-feature component-area-feature projection-profile-feature feature comparator feature-factory curve-approximation otap-common)

# set(ALL_SERIAL_FEATURES profile-projection upper-projection lower-projection upper-word-profile lower-word-profile background-transitions contour-line-pair contour-single-line shape-context sift surf freak orb rath-2003 chamfer)

# set(ALL_FEATURES aspect-ratio component-area ${ALL_SERIAL_FEATURES})

# option(TRACE_CHECK_POINTS "Convert CPx to Trace Points in the code" OFF)
# option(USE_ASPECT_RATIO "Use Aspect Ratio as a default feature" OFF)
# option(USE_LINE_PAIR "Use Line Pair as a default feature" OFF)
# option(USE_SINGLE_LINE "Use Single Line as a default feature" OFF)
# option(USE_SHAPE_CONTEXT "Use Shape Context as a default feature" OFF)

# set(DO_CC_MATRIX_TESTS 1)
# set(DO_WS_MATRIX_TESTS 0)
# set(DO_OTAP_WS_OTM20_TEST 0)
# set(DO_OTAP_WS_OTM5_TEST 0)
# set(DO_OTAP_WS_GW20_TEST 0)
# set(DO_OTAP_WS_CONTEST_TEST 1)
# set(DO_IMAGE_RETRIEVAL_GW20_TESTS 0)
# set(DO_IMAGE_RETRIEVAL_LM_TESTS 0)

# set(RUN_MEMCHECK 0)
# set(RUN_PROFILER 0)

# set(DATASET_DIR ${PROJECT_SOURCE_DIR}/../datasets)

# set(DATASET_IAM_GW iam-gw)
# set(DATASET_IAM_GW_1472 iam-gw-1472)
# set(DATASET_IAM_GW_77 iam-gw-77)
# set(DATASET_IAM_GW_16_I4586 iam-gw-16-i4586)
# set(DATASET_IAM_GW_16_I928 iam-gw-16-i928)
# set(DATASET_IAM_GW_02 iam-gw-02)
# set(DATASET_IAM_GW_WORDS iam-gw-words)
# set(DATASET_LM_COMPONENTS_20K leylamecnun-components-20k)
# set(DATASET_LM_38 leylamecnun-38)
# set(DATASET_LM_1229 leylamecnun-1229)
# set(DATASET_LM_5_I300 leylamecnun-5-i300)
# set(DATASET_LM_LINES leylamecnun-lines)
# set(DATASET_LM_WORDS leylamecnun-words)
# set(DATASET_LM_COMPONENT_LIB leylamecnun-cc-library)
# set(DATASET_LM_PAGES leylamecnun-pages)
# set(DATASET_IFNENIT_5_I300 ifnenit-a-5-i300)
# set(DATASET_IFNENIT_44_I1000 ifnenit-a-44-i1000)
# set(DATASET_IFNENIT_822 ifnenit-a-822)
# set(DATASET_KAGGLE_GENDER_HW_COMP kaggle-gender-hw-components)
# set(DATASET_KUFIC_COMPONENTS kufic-clean-components-combined)
# set(DATASET_DEBUG_TEST debug-test)

# set(RESULT_DIR ${PROJECT_SOURCE_DIR}/../experiments)

# configure_file(cmake-options.hpp.in ${PROJECT_SOURCE_DIR}/otap-word-spot/cmake-options.hpp)

## global tags for all dependencies etc

# set_source_files_properties( GTAGS PROPERTIES GENERATED true)
# add_custom_command ( OUTPUT GTAGS
#    COMMAND gtags
#    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/../ )

# TEST TARGETS

# add_executable(cpptest cpptest.cpp)

# target_link_libraries(cpptest)


# add_executable(otap-common-test otap-common-test otap-common)
# add_executable(csv-reader-test csv-reader-test csv-reader otap-common)

# add_executable(feature-test
#   feature-test
#   feature
#   otap-common
#   test-common-defs)
  
# add_executable(comparator-test
#   comparator-test
#   comparator
#   feature
#   curve-approximation
#   otap-common
#   test-common-defs)

# add_executable(simple-comparator-test
#   simple-comparator-test
#   simple-comparator
#   otap-common
#   test-common-defs)

# add_executable(otap-calc-dist-test
#   otap-calc-dist-test
#   otap-calc-dist.hpp
#   csv-reader
#   simple-comparator
#   curve-approximation
#   otap-common
#   test-common-defs)

# add_executable(aggregate-comparator-test
#   aggregate-comparator-test
#   aggregate-comparator
#   comparator
#   feature
#   curve-approximation
#   otap-common
#   test-common-defs)

# add_executable(line-pair-feature-test
#   line-pair-feature-test
#   line-pair-feature
#   feature
#   comparator
#   otap-common 
#   test-common-defs)

# add_executable(single-line-feature-test
#   single-line-feature-test
#   single-line-feature
#   feature
#   comparator
#   otap-common 
#   test-common-defs)

# add_executable(shape-context-feature-test
#   shape-context-feature-test
#   shape-context-feature
#   feature
#   comparator
#   otap-common 
#   test-common-defs)

# add_executable(sift-feature-test
#   sift-feature-test
#   ${CC_FILES}
#   test-common-defs)

# add_executable(surf-feature-test
#   surf-feature-test
#   ${CC_FILES}
#   test-common-defs)

# add_executable(freak-feature-test
#   freak-feature-test
#   ${CC_FILES}
#   test-common-defs)

# add_executable(chamfer-feature-test
#   chamfer-feature-test
#   chamfer-feature
#   surf-feature
#   sift-feature
#   freak-feature
#   feature
#   comparator
#   connected-component
#   otap-common 
#   test-common-defs)

# add_executable(component-area-feature-test
#   component-area-feature-test
#   component-area-feature
#   surf-feature
#   sift-feature
#   freak-feature
#   feature
#   comparator
#   connected-component
#   otap-common 
#   test-common-defs)

# add_executable(projection-profile-feature-test
#   projection-profile-feature-test
#   ${CC_FILES}
#   test-common-defs)

# add_executable(connected-component-test 
#   connected-component-test
#   test-common-defs 
#   ${CC_FILES})

# add_executable(component-filter-test
#   component-filter-test
#   component-filter
#   test-common-defs 
#   ${CC_FILES})

# add_executable(connected-component-library-test 
#   connected-component-library-test
#   component-filter
#   test-common-defs 
#   ${CC_FILES})


# add_executable(page-image-test 
#   page-image-test 
#   page-image 
#   test-common-defs
#   ${CC_FILES})

# add_executable(
# page-graph-test
#   page-graph-test
#   page-graph
#   page-image 
#   test-common-defs
#   ${CC_FILES})


# add_executable(curve-approximation-test
#   curve-approximation-test
#   curve-approximation
#   page-image
#   test-common-defs
#   ${CC_FILES})

# add_executable(word-spot-test 
#   word-spot-test 
#   word-spot 
#   page-image
#   test-common-defs
#   ${CC_FILES})

# add_executable(word-spot-page-query-test 
#   word-spot-page-query-test 
#   word-spot-page-query 
#   word-spot
#   page-image
#   ${CC_FILES}
#   test-common-defs)

# add_executable(word-spot-query-test 
#   word-spot-query-test 
#   word-spot-query 
#   word-spot-page-query
#   word-spot-page-query
#   word-spot
#   page-image
#   test-common-defs
#   ${CC_FILES})

# add_executable(word-spot-query-manager-test 
#   word-spot-query-manager-test 
#   word-spot-query-manager 
#   word-spot-query
#   word-spot-page-query
#   word-spot
#   page-image
#   ${CC_FILES}
#   test-common-defs)

# # MAIN TARGETS

# add_executable(similar-files
#   similar-files
#   word-spot-page-query
#   word-spot-query
#   word-spot
#   page-image
#   ${CC_FILES})

# add_executable(similar-matches
#   similar-matches
#   page-graph
#   page-image
#   ${CC_FILES})

# add_executable(cc-matrix
#   cc-matrix
#   curve-approximation
#   page-image
#   ${CC_FILES})
  
# add_executable(ws-matrix
#   ws-matrix
#   word-spot-query
#   word-spot-page-query
#   word-spot
#   page-image
#   ${CC_FILES})

# add_executable(image-retrieval
#   image-retrieval
#   word-spot
#   page-image
#   ${CC_FILES})

# add_executable(retrieval-recall-precision
#   retrieval-recall-precision
#   word-spot
#   page-image
#   ${CC_FILES})

# add_executable(shard-page-components
#   shard-page-components
#   page-image
#   ${CC_FILES})

# add_executable(otap-calc-dist
#   otap-calc-dist.cpp
#   otap-calc-dist.hpp
#   simple-comparator
#   csv-reader
#   otap-common)

# add_executable(otap-word-spot
#   main
#   word-spot-query-manager 
#   word-spot-query
#   word-spot-page-query
#   word-spot
#   page-image
#   ${CC_FILES})  

# add_library(otapy SHARED
#   otap-python
#   page-image
#   ${CC_FILES})

# LINK DEFINITIONS

# target_link_libraries(abc opencv_core opencv_highgui opencv_imgproc boost_program_options)

# target_link_libraries(otap-common-test ${OTAP_TESTING_LIBS})

# target_link_libraries(otap-calc-dist-test ${OTAP_TESTING_LIBS})

# target_link_libraries(csv-reader-test ${OTAP_TESTING_LIBS})

# target_link_libraries(feature-test ${OTAP_TESTING_LIBS})

# target_link_libraries(comparator-test ${OTAP_TESTING_LIBS})

# target_link_libraries(simple-comparator-test ${OTAP_TESTING_LIBS})

# # target_link_libraries(aggregate-comparator-test ${OTAP_TESTING_LIBS})

# target_link_libraries(line-pair-feature-test ${OTAP_TESTING_LIBS})

# target_link_libraries(single-line-feature-test ${OTAP_TESTING_LIBS})

# target_link_libraries(shape-context-feature-test ${OTAP_TESTING_LIBS})

# target_link_libraries(sift-feature-test ${OTAP_TESTING_LIBS})

# target_link_libraries(surf-feature-test ${OTAP_TESTING_LIBS})

# target_link_libraries(chamfer-feature-test ${OTAP_TESTING_LIBS})

# target_link_libraries(component-area-feature-test ${OTAP_TESTING_LIBS})

# target_link_libraries(freak-feature-test ${OTAP_TESTING_LIBS})

# target_link_libraries(projection-profile-feature-test ${OTAP_TESTING_LIBS})

# target_link_libraries(connected-component-test ${OTAP_TESTING_LIBS})

# target_link_libraries(connected-component-library-test ${OTAP_TESTING_LIBS})

# target_link_libraries(component-filter-test ${OTAP_TESTING_LIBS})

# target_link_libraries(page-image-test ${OTAP_TESTING_LIBS})

# target_link_libraries(page-graph-test ${OTAP_TESTING_LIBS})

# target_link_libraries(curve-approximation-test ${OTAP_TESTING_LIBS})

# target_link_libraries(word-spot-test ${OTAP_TESTING_LIBS})

# target_link_libraries(word-spot-page-query-test ${OTAP_TESTING_LIBS})

# target_link_libraries(word-spot-query-test ${OTAP_TESTING_LIBS})

# target_link_libraries(word-spot-query-manager-test ${OTAP_TESTING_LIBS})

# target_link_libraries(otap-word-spot ${OTAP_COMMON_LIBS})

# target_link_libraries(otapy ${OTAP_COMMON_LIBS})

# target_link_libraries(cc-matrix ${OTAP_TESTING_LIBS})

# target_link_libraries(ws-matrix ${OTAP_TESTING_LIBS})

# target_link_libraries(image-retrieval ${OTAP_TESTING_LIBS})

# target_link_libraries(retrieval-recall-precision ${OTAP_TESTING_LIBS})

# target_link_libraries(shard-page-components ${OTAP_COMMON_LIBS})

# target_link_libraries(otap-calc-dist ${OTAP_COMMON_LIBS})

# target_link_libraries(similar-files ${OTAP_COMMON_LIBS})

# target_link_libraries(similar-matches ${OTAP_COMMON_LIBS})

# add_custom_command(TARGET abc
#   POST_BUILD
#   COMMAND abc ../test-image.png)

# add_custom_command(TARGET cppdeneme
#   POST_BUILD
#   COMMAND cppdeneme)

# add_custom_command(TARGET otap-common-test
#   POST_BUILD
#   COMMAND otap-common-test)

# add_custom_command(TARGET otap-calc-dist-test
#   POST_BUILD
#   COMMAND otap-calc-dist-test)

# add_custom_command(TARGET csv-reader-test
#   POST_BUILD
#   COMMAND csv-reader-test)

# add_custom_command(TARGET feature-test
#   POST_BUILD
#   COMMAND feature-test)

# add_custom_command(TARGET comparator-test
#   POST_BUILD
#   COMMAND comparator-test)

# add_custom_command(TARGET simple-comparator-test
#   POST_BUILD
#   COMMAND simple-comparator-test)


# # add_custom_command(TARGET aggregate-comparator-test
# #   POST_BUILD
# #   COMMAND aggregate-comparator-test)

# add_custom_command(TARGET line-pair-feature-test
#    POST_BUILD
#    COMMAND line-pair-feature-test)

# add_custom_command(TARGET single-line-feature-test
#    POST_BUILD
#    COMMAND single-line-feature-test)

# add_custom_command(TARGET shape-context-feature-test
#    POST_BUILD
#    COMMAND shape-context-feature-test)


# add_custom_command(TARGET sift-feature-test
#    POST_BUILD
#    COMMAND sift-feature-test)

# add_custom_command(TARGET surf-feature-test
#    POST_BUILD
#    COMMAND surf-feature-test)

# add_custom_command(TARGET chamfer-feature-test
#    POST_BUILD
#    COMMAND chamfer-feature-test)

# add_custom_command(TARGET component-area-feature-test
#    POST_BUILD
#    COMMAND component-area-feature-test)

# add_custom_command(TARGET projection-profile-feature-test
#    POST_BUILD
#    COMMAND projection-profile-feature-test)

# add_custom_command(TARGET freak-feature-test
#    POST_BUILD
#    COMMAND freak-feature-test)

# add_custom_command(TARGET connected-component-test
#   POST_BUILD
#   COMMAND connected-component-test)

# add_custom_command(TARGET connected-component-library-test
#   POST_BUILD
#   COMMAND connected-component-library-test)

# add_custom_command(TARGET component-filter-test
#   POST_BUILD
#   COMMAND component-filter-test)

# add_custom_command(TARGET page-image-test
#    POST_BUILD
#    COMMAND page-image-test)

# add_custom_command(TARGET page-graph-test
#    POST_BUILD
#    COMMAND page-graph-test)

# add_custom_command(TARGET curve-approximation-test
#    POST_BUILD
#    COMMAND curve-approximation-test)

# add_custom_command(TARGET word-spot-test
#   POST_BUILD
#   COMMAND word-spot-test)

# add_custom_command(TARGET word-spot-page-query-test
#   POST_BUILD
#   COMMAND word-spot-page-query-test)

# add_custom_command(TARGET word-spot-query-test
#   POST_BUILD
#   COMMAND word-spot-query-test)

# add_custom_command(TARGET word-spot-query-manager-test
#   POST_BUILD
#   COMMAND word-spot-query-manager-test)

# # add_custom_command(TARGET cc-matrix
# #   POST_BUILD
# #   COMMAND valgrind --tool=memcheck --leak-check=yes ./cc-matrix --result-size 100 --input ../../../experiments/two/2.png --output-file cc-matrix-result.yml)

# if(${DO_OTAP_WS_OTM20_TEST})
#   set (OTAP_WS_PAGE_DIR ${PROJECT_SOURCE_DIR}/../web/static/wordspot/pages20/)
#   file(GLOB OTAP_WS_QUERY_FILES ${PROJECT_SOURCE_DIR}/../experiments/otap-word-spot/otm20/queries/*.png)
#   foreach(ff ${OTAP_WS_QUERY_FILES})
#     message (${ff})
#   endforeach(ff)
#   set (OTAP_WS_OUTPUT_DIR ${PROJECT_SOURCE_DIR}/../experiments/otap-word-spot/otm20/results/)
# #  set (OTAP_WS_FEATURES contour-line-pair contour-single-line shape-context)
#   set (OTAP_WS_FEATURES shape-context)
# #  set (OTAP_WS_SERIAL_COMPARATORS minimums in-order spanning-minimums)
#   set (OTAP_WS_SERIAL_COMPARATORS minimums minimums in-order spanning-minimums)

#   set (OTAP_WS_SCALAR_COMPARATORS plain)
  
#   foreach(query_file ${OTAP_WS_QUERY_FILES})
#     string(RANDOM outdir_name)
#     add_custom_command(TARGET otap-word-spot
#       POST_BUILD
#       COMMAND mkdir -p ${OTAP_WS_OUTPUT_DIR}/${outdir_name})
#     add_custom_command(TARGET otap-word-spot
#       POST_BUILD
#       COMMAND ${PROJECT_SOURCE_DIR}/../scripts/2colorthresh ${query_file} ${OTAP_WS_OUTPUT_DIR}${outdir_name}/query.png)

#     foreach(feature ${OTAP_WS_FEATURES})
#       foreach(serial_cmp ${OTAP_WS_SERIAL_COMPARATORS})
# 	foreach(scalar_cmp ${OTAP_WS_SCALAR_COMPARATORS})
# 	  add_custom_command(TARGET otap-word-spot
# 	    POST_BUILD
# 	    COMMAND mkdir -p ${OTAP_WS_OUTPUT_DIR}/${outdir_name}/${feature}-${serial_cmp}-${scalar_cmp})
# 	  add_custom_command(TARGET otap-word-spot
# 	    POST_BUILD
# 	    COMMAND otap-word-spot --result-size 20 --image-dir ${OTAP_WS_PAGE_DIR} --query-dir ${OTAP_WS_OUTPUT_DIR}/${outdir_name}/query.png --output-filename ${OTAP_WS_OUTPUT_DIR}/${outdir_name}/${feature}-${serial_cmp}-${scalar_cmp}/result.html --feature ${feature} --serial-comparator ${serial_cmp} --scalar-comparator ${scalar_cmp} --span-size 5)
# 	endforeach(scalar_cmp)
#       endforeach(serial_cmp)
#     endforeach(feature)
#   endforeach(query_file)
# endif()




# if(${DO_OTAP_WS_OTM5_TEST})
#   set (OTAP_WS_PAGE_DIR ${PROJECT_SOURCE_DIR}/../web/static/wordspot/pages/)
#   file(GLOB OTAP_WS_QUERY_FILES ${PROJECT_SOURCE_DIR}/../experiments/otap-word-spot/otm5/queries/*.png)
#   foreach(ff ${OTAP_WS_QUERY_FILES})
#     message (${ff})
#   endforeach(ff)
#   set (OTAP_WS_OUTPUT_DIR ${PROJECT_SOURCE_DIR}/../experiments/otap-word-spot/otm5/results/)
# #  set (OTAP_WS_FEATURES contour-line-pair contour-single-line shape-context)
#   set (OTAP_WS_FEATURES contour-single-line shape-context)
# #  set (OTAP_WS_SERIAL_COMPARATORS minimums in-order spanning-minimums)
#   set (OTAP_WS_SERIAL_COMPARATORS minimums)

#   set (OTAP_WS_SCALAR_COMPARATORS plain)
  
#   foreach(query_file ${OTAP_WS_QUERY_FILES})
#     string(RANDOM outdir_name)
#     add_custom_command(TARGET otap-word-spot
#       POST_BUILD
#       COMMAND mkdir -p ${OTAP_WS_OUTPUT_DIR}/${outdir_name})
#     add_custom_command(TARGET otap-word-spot
#       POST_BUILD
#       COMMAND ${PROJECT_SOURCE_DIR}/../scripts/2colorthresh ${query_file} ${OTAP_WS_OUTPUT_DIR}${outdir_name}/query.png)

#     foreach(feature ${OTAP_WS_FEATURES})
#       foreach(serial_cmp ${OTAP_WS_SERIAL_COMPARATORS})
# 	foreach(scalar_cmp ${OTAP_WS_SCALAR_COMPARATORS})
# 	  add_custom_command(TARGET otap-word-spot
# 	    POST_BUILD
# 	    COMMAND mkdir -p ${OTAP_WS_OUTPUT_DIR}/${outdir_name}/${feature}-${serial_cmp}-${scalar_cmp})
# 	  add_custom_command(TARGET otap-word-spot
# 	    POST_BUILD
# 	    COMMAND otap-word-spot --result-size 20 --image-dir ${OTAP_WS_PAGE_DIR} --query-dir ${OTAP_WS_OUTPUT_DIR}/${outdir_name}/query.png --output-filename ${OTAP_WS_OUTPUT_DIR}/${outdir_name}/${feature}-${serial_cmp}-${scalar_cmp}/result.html --feature ${feature} --serial-comparator ${serial_cmp} --scalar-comparator ${scalar_cmp})
# 	endforeach(scalar_cmp)
#       endforeach(serial_cmp)
#     endforeach(feature)
#   endforeach(query_file)
# endif()

# if(${DO_OTAP_WS_GW20_TEST})
#   set (OTAP_WS_PAGE_DIR ../../../experiments/iam-gw/gw/data/line_images_normalized/)
#   file(GLOB OTAP_WS_QUERY_FILES ../../experiments/iam-gw/gw/queries/*.png)
#   foreach(ff ${OTAP_WS_QUERY_FILES})
#     message (${ff})
#   endforeach(ff)
#   set (OTAP_WS_OUTPUT_DIR ../../../experiments/iam-gw/gw/results/)
# #  set (OTAP_WS_FEATURES contour-line-pair contour-single-line shape-context)
#   set (OTAP_WS_FEATURES shape-context)
# #  set (OTAP_WS_SERIAL_COMPARATORS minimums in-order spanning-minimums)
#   set (OTAP_WS_SERIAL_COMPARATORS minimums)
#   set (OTAP_WS_SCALAR_COMPARATORS plain)
  
#   foreach(query_file ${OTAP_WS_QUERY_FILES})
#     string(RANDOM outdir_name)
#      add_custom_command(TARGET otap-word-spot
#        POST_BUILD
#        COMMAND mkdir -p ${OTAP_WS_OUTPUT_DIR}/${outdir_name} && cp ${query_file} ${OTAP_WS_OUTPUT_DIR}/${outdir_name}/query.png)
#     foreach(feature ${OTAP_WS_FEATURES})
#       foreach(serial_cmp ${OTAP_WS_SERIAL_COMPARATORS})
# 	foreach(scalar_cmp ${OTAP_WS_SCALAR_COMPARATORS})
# 	  add_custom_command(TARGET otap-word-spot
# 	    POST_BUILD
# 	    COMMAND mkdir -p ${OTAP_WS_OUTPUT_DIR}/${outdir_name}/${feature}-${serial_cmp}-${scalar_cmp})
# 	  add_custom_command(TARGET otap-word-spot
# 	    POST_BUILD
# 	    COMMAND otap-word-spot --result-size 20 --image-dir ${OTAP_WS_PAGE_DIR} --query-dir ${OTAP_WS_OUTPUT_DIR}/${outdir_name}/query.png --output-filename ${OTAP_WS_OUTPUT_DIR}/${outdir_name}/${feature}-${serial_cmp}-${scalar_cmp}/result.html --feature ${feature} --serial-comparator ${serial_cmp} --scalar-comparator ${scalar_cmp} > ${OTAP_WS_OUTPUT_DIR}/${outdir_name}/output.log)
# 	endforeach(scalar_cmp)
#       endforeach(serial_cmp)
#     endforeach(feature)
#   endforeach(query_file)
# endif()

# if(${DO_OTAP_WS_CONTEST_TEST})
#   set (OTAP_WS_PAGE_DIR ../../../datasets/wordspot-contest/images/)
#   file(GLOB OTAP_WS_QUERY_DIRS ../../datasets/wordspot-contest/Q*/)

#   # message(=====INPUT FILES=====)
#   # foreach(ff ${OTAP_WS_PAGE_FILES})
#   #   message (${ff})
#   # endforeach(ff)

#   message(=====QUERY FILES=====)
#   foreach(ff ${OTAP_WS_QUERY_DIRS})
#     message (${ff})
#   endforeach(ff)

#   set (OTAP_WS_OUTPUT_DIR ../../../experiments/wordspot-contest/results/)
# #  set (OTAP_WS_FEATURES contour-line-pair contour-single-line shape-context)
# #  set (OTAP_WS_FEATURES projection-profile upper-projection contour-single-line rath-2003 shape-context)
# # set (OTAP_WS_FEATURES ${ALL_SERIAL_FEATURES})
# set (OTAP_WS_FEATURES projection-profile)

# #  set (OTAP_WS_SERIAL_COMPARATORS minimums in-order spanning-minimums)
#   set (OTAP_WS_SERIAL_COMPARATORS dynamic-time-warping-normalized)
#   set (OTAP_WS_SCALAR_COMPARATORS plain)
  
#   foreach(query_dir ${OTAP_WS_QUERY_DIRS})
# #    file(GLOB OTAP_WS_QUERY_FILES ${query_dir}/*.tif)
#     foreach(query_file ${query_dir})
#     string(RANDOM random_dir)
#     foreach(feature ${OTAP_WS_FEATURES})
#     foreach(serial_cmp ${OTAP_WS_SERIAL_COMPARATORS})
# 	  foreach(scalar_cmp ${OTAP_WS_SCALAR_COMPARATORS})
#         string(REGEX REPLACE "^(.*)/([^/]+)/([^./]+)(\\.[^./]+)?$" "\\2-\\3" query_file_out ${query_file})
#         message(${query_file_out})
#         set (outfile ${OTAP_WS_OUTPUT_DIR}/${query_file_out}/${feature}-${serial_cmp}-${scalar_cmp}/${random_dir}/)
#         message(${outfile})
# 	      add_custom_command(TARGET otap-word-spot
# 	        POST_BUILD
# 	        COMMAND mkdir -p ${outfile})
#           set(the_params --result-size 50 --input ${OTAP_WS_PAGE_DIR} -q ${query_file} --output-file ${outfile} --feature ${feature} --serial-comparator ${serial_cmp} --scalar-comparator ${scalar_cmp} --wordspot-use-feature-pool --use-multithreading )
# 	      add_custom_command(TARGET otap-word-spot
# 	        POST_BUILD
# 	        COMMAND echo ${the_params})
# 	      add_custom_command(TARGET otap-word-spot
# 	        POST_BUILD
# 	        COMMAND otap-word-spot ${the_params} > ${outfile}/output.log)

# 	     endforeach(scalar_cmp)
#        endforeach(serial_cmp)
#      endforeach(feature)
#    endforeach(query_file)
#  endforeach(query_dir)
# endif()
 




# if(${DO_CC_MATRIX_TESTS})

#   set (CC_MATRIX_PAGE_IMAGE ../../../experiments/three/page.png)
#   set (CC_MATRIX_QUERY_IMAGE ../../../experiments/three/query.png)
#   set (CC_MATRIX_OUTPUT_DIR ../../../experiments/three/)
#   set (CC_MATRIX_FEATURES ${ALL_FEATURES})
#   set (CC_MATRIX_RICH_FEATURES sift freak surf)
#   #  set (CC_MATRIX_SERIAL_COMPARATORS minimums maximums in-order spanning-minimums spanning-maximums)
#   set (CC_MATRIX_SERIAL_COMPARATORS minimums dynamic-minimums)
#   # set (CC_MATRIX_SCALAR_COMPARATORS plain sigmoid sigmoid-square)
#   set (CC_MATRIX_SCALAR_COMPARATORS plain)

#   foreach(feature ${CC_MATRIX_RICH_FEATURES})
#     foreach(serial_cmp ${CC_MATRIX_SERIAL_COMPARATORS})
#       foreach(scalar_cmp ${CC_MATRIX_SCALAR_COMPARATORS})
# 	foreach(desc_size 1 2 4 9)
# 	add_custom_command(TARGET cc-matrix
# 	  POST_BUILD
# 	  COMMAND echo Processing: ${feature}-K${desc_size}-${serial_cmp}-${scalar_cmp}-)
# 	add_custom_command(TARGET cc-matrix
# 	  POST_BUILD
# 	  COMMAND mkdir -p ${CC_MATRIX_OUTPUT_DIR}/${feature}-K${desc_size}-${serial_cmp}-${scalar_cmp})
# 	add_custom_command(TARGET cc-matrix
# 	  POST_BUILD
# 	  COMMAND cc-matrix --result-size 100 --input ${CC_MATRIX_PAGE_IMAGE} --output-file ${CC_MATRIX_OUTPUT_DIR}/${feature}-K${desc_size}-${serial_cmp}-${scalar_cmp}/${feature}-K${desc_size}-${serial_cmp}-${scalar_cmp}.html --descriptor-size ${desc_size} --feature ${feature} --serial-comparator ${serial_cmp} --scalar-comparator ${scalar_cmp} > ${CC_MATRIX_OUTPUT_DIR}/${feature}-K${desc_size}-${serial_cmp}-${scalar_cmp}/${feature}-K${desc_size}-${serial_cmp}-${scalar_cmp}.log)
# 	endforeach(desc_size)
#       endforeach(scalar_cmp)
#     endforeach(serial_cmp)
#   endforeach(feature)

#   foreach(feature ${CC_MATRIX_FEATURES})
#     foreach(serial_cmp ${CC_MATRIX_SERIAL_COMPARATORS})
#       foreach(scalar_cmp ${CC_MATRIX_SCALAR_COMPARATORS})
# 	add_custom_command(TARGET cc-matrix
# 	  POST_BUILD
# 	  COMMAND echo Processing: ${feature}-${serial_cmp}-${scalar_cmp})
# 	add_custom_command(TARGET cc-matrix
# 	  POST_BUILD
# 	  COMMAND mkdir -p ${CC_MATRIX_OUTPUT_DIR}/${feature}-${serial_cmp}-${scalar_cmp})
# 	add_custom_command(TARGET cc-matrix
# 	  POST_BUILD
# 	  COMMAND cc-matrix --result-size 100 --input ${CC_MATRIX_PAGE_IMAGE} --output-file ${CC_MATRIX_OUTPUT_DIR}/${feature}-${serial_cmp}-${scalar_cmp}/${feature}-${serial_cmp}-${scalar_cmp}.html --feature ${feature} --serial-comparator ${serial_cmp} --scalar-comparator ${scalar_cmp} > ${CC_MATRIX_OUTPUT_DIR}/${feature}-${serial_cmp}-${scalar_cmp}/${feature}-${serial_cmp}-${scalar_cmp}.log)
#       endforeach(scalar_cmp)
#     endforeach(serial_cmp)
#   endforeach(feature)

# endif()


# if(${DO_IMAGE_RETRIEVAL_GW20_TESTS})
#   set (GW20_WORDS ../../../experiments/iam-gw/gw/data/word_images_normalized/)
#   file(GLOB IMAGE_RETRIEVAL_QUERY_FILES ../../experiments/iam-gw/image-retrieval/queries/*.png)
#   foreach(ff ${IMAGE_RETRIEVAL_QUERY_FILES})
#     message (${ff})
#   endforeach(ff)

#   set (IMAGE_RETRIEVAL_OUTPUT_DIR ../../../experiments/iam-gw/image-retrieval/results/)
# #  set (IMAGE_RETRIEVAL_FEATURES contour-line-pair contour-single-line shape-context)
#   set (IMAGE_RETRIEVAL_FEATURES chamfer freak sift surf contour-line-pair contour-single-line shape-context)
# #  set (IMAGE_RETRIEVAL_SERIAL_COMPARATORS minimums in-order spanning-minimums)
#   set (IMAGE_RETRIEVAL_SERIAL_COMPARATORS dynamic-minimums)
#   set (IMAGE_RETRIEVAL_SCALAR_COMPARATORS euclidean)
#   foreach(query_file ${IMAGE_RETRIEVAL_QUERY_FILES})
#     string(RANDOM outdir_name)
#      add_custom_command(TARGET image-retrieval
#        POST_BUILD
#        COMMAND mkdir -p ${IMAGE_RETRIEVAL_OUTPUT_DIR}/${outdir_name} && cp ${query_file} ${IMAGE_RETRIEVAL_OUTPUT_DIR}/${outdir_name}/)
#     foreach(feature ${IMAGE_RETRIEVAL_FEATURES})
#       foreach(serial_cmp ${IMAGE_RETRIEVAL_SERIAL_COMPARATORS})
# 	foreach(scalar_cmp ${IMAGE_RETRIEVAL_SCALAR_COMPARATORS})
# 	  add_custom_command(TARGET image-retrieval 
# 	    POST_BUILD
# 	    COMMAND echo Processing: ${outdir_name}/${feature}-${serial_cmp}-${scalar_cmp})
# 	  add_custom_command(TARGET image-retrieval
# 	    POST_BUILD
# 	    COMMAND mkdir -p ${IMAGE_RETRIEVAL_OUTPUT_DIR}/${outdir_name}/${feature}-${serial_cmp}-${scalar_cmp})
# 	  add_custom_command(TARGET image-retrieval
# 	    POST_BUILD
# 	    COMMAND image-retrieval --result-size 20 --input ${GW20_WORDS} --query-image ${query_file} --output-file ${IMAGE_RETRIEVAL_OUTPUT_DIR}/${outdir_name}/${feature}-${serial_cmp}-${scalar_cmp}/result.html --feature ${feature} --serial-comparator ${serial_cmp} --scalar-comparator ${scalar_cmp} > ${IMAGE_RETRIEVAL_OUTPUT_DIR}/${outdir_name}/output.log)
# 	endforeach(scalar_cmp)
#       endforeach(serial_cmp)
#     endforeach(feature)
#   endforeach(query_file)
# endif()


# if(${DO_IMAGE_RETRIEVAL_LM_TESTS})
#   set (LM_WORDS ../../../experiments/leylamecnun/words/)
#   file(GLOB IMAGE_RETRIEVAL_QUERY_FILES ../../experiments/leylamecnun/queries/*.png)
#   foreach(ff ${IMAGE_RETRIEVAL_QUERY_FILES})
#     message (${ff})
#   endforeach(ff)

#   set (IMAGE_RETRIEVAL_OUTPUT_DIR ../../../experiments/leylamecnun/results/)
# #  set (IMAGE_RETRIEVAL_FEATURES contour-line-pair contour-single-line shape-context)
#   set (IMAGE_RETRIEVAL_FEATURES component-area chamfer freak sift surf contour-line-pair contour-single-line shape-context)
# #  set (IMAGE_RETRIEVAL_SERIAL_COMPARATORS minimums in-order spanning-minimums)
#   set (IMAGE_RETRIEVAL_SERIAL_COMPARATORS dynamic-minimums)
#   set (IMAGE_RETRIEVAL_SCALAR_COMPARATORS euclidean)
#   foreach(query_file ${IMAGE_RETRIEVAL_QUERY_FILES})
#     string(RANDOM outdir_name)
#      add_custom_command(TARGET image-retrieval
#        POST_BUILD
#        COMMAND mkdir -p ${IMAGE_RETRIEVAL_OUTPUT_DIR}/${outdir_name} && cp ${query_file} ${IMAGE_RETRIEVAL_OUTPUT_DIR}/${outdir_name}/)
#     foreach(feature ${IMAGE_RETRIEVAL_FEATURES})
#       foreach(serial_cmp ${IMAGE_RETRIEVAL_SERIAL_COMPARATORS})
# 	foreach(scalar_cmp ${IMAGE_RETRIEVAL_SCALAR_COMPARATORS})
# 	  add_custom_command(TARGET image-retrieval 
# 	    POST_BUILD
# 	    COMMAND echo Processing: ${outdir_name}/${feature}-${serial_cmp}-${scalar_cmp})
# 	  add_custom_command(TARGET image-retrieval
# 	    POST_BUILD
# 	    COMMAND mkdir -p ${IMAGE_RETRIEVAL_OUTPUT_DIR}/${outdir_name}/${feature}-${serial_cmp}-${scalar_cmp})
# 	  add_custom_command(TARGET image-retrieval
# 	    POST_BUILD
# 	    COMMAND image-retrieval --result-size 20 --input ${LM_WORDS} --query-image ${query_file} --output-file ${IMAGE_RETRIEVAL_OUTPUT_DIR}/${outdir_name}/${feature}-${serial_cmp}-${scalar_cmp}/result.html --feature ${feature} --serial-comparator ${serial_cmp} --scalar-comparator ${scalar_cmp} > ${IMAGE_RETRIEVAL_OUTPUT_DIR}/${outdir_name}/output.log)
# 	endforeach(scalar_cmp)
#       endforeach(serial_cmp)
#     endforeach(feature)
#   endforeach(query_file)
# endif()

# if(${DO_WS_MATRIX_TESTS})

#   set (WS_MATRIX_PAGE_IMAGE ../../../experiments/ws/three/page.png)
#   set (WS_MATRIX_QUERY_IMAGE ../../../experiments/ws/three/query.png)
#   set (WS_MATRIX_OUTPUT_DIR ../../../experiments/ws/three/)
#   set (WS_MATRIX_FEATURES aspect-ratio contour-line-pair contour-single-line shape-context)
#   #  set (WS_MATRIX_SERIAL_COMPARATORS minimums maximums in-order spanning-minimums spanning-maximums)
#   set (WS_MATRIX_SERIAL_COMPARATORS minimums)
#   # set (WS_MATRIX_SCALAR_COMPARATORS plain sigmoid sigmoid-square)
#   set (WS_MATRIX_SCALAR_COMPARATORS plain)
  
#   foreach(feature ${WS_MATRIX_FEATURES})
#     foreach(serial_cmp ${WS_MATRIX_SERIAL_COMPARATORS})
#       foreach(scalar_cmp ${WS_MATRIX_SCALAR_COMPARATORS})
# 	add_custom_command(TARGET ws-matrix
# 	  POST_BUILD
# 	  COMMAND mkdir -p ${WS_MATRIX_OUTPUT_DIR}/${feature}-${serial_cmp}-${scalar_cmp})
# 	add_custom_command(TARGET ws-matrix
# 	  POST_BUILD
# 	  COMMAND ws-matrix --result-size 100 --page-image ${WS_MATRIX_PAGE_IMAGE} --query-image ${WS_MATRIX_QUERY_IMAGE} --output-file ${WS_MATRIX_OUTPUT_DIR}/${feature}-${serial_cmp}-${scalar_cmp}/${feature}-${serial_cmp}-${scalar_cmp}.html --feature ${feature} --serial-comparator ${serial_cmp} --scalar-comparator ${scalar_cmp})
#       endforeach(scalar_cmp)
#     endforeach(serial_cmp)
#   endforeach(feature)
# endif()


# macro(run_recall_precision)

#   add_custom_command(TARGET retrieval-recall-precision
#     POST_BUILD
#     COMMAND mkdir -p ${RP_OUTPUT_DIR}/words/)

#   string(REGEX REPLACE " +" "-" extra_opts_filename ${RP_EXTRA_OPTIONS})

#   foreach(feature ${RP_FEATURES})
#     foreach(serial_cmp ${RP_SERIAL_COMPARATORS})
#       foreach(scalar_cmp ${RP_SCALAR_COMPARATORS})
# 	foreach(desc_size ${RP_DESC_SIZE})
# 	  foreach(subst_cost_func ${RP_SUBST_FUNC})
# 	    foreach(subst_cost_mod ${RP_SUBST_MOD})
# 	      foreach(delete_cost_func ${RP_DELETE_FUNC})
# 		foreach(delete_cost_mod ${RP_DELETE_MOD})
# 		  foreach(location_weight ${RP_LOCATION_WEIGHTS})
# 		    string(RANDOM outfile_suffix)

# 		    set(command_opts --feature ${feature} --descriptor-size ${desc_size} --serial-comparator ${serial_cmp} --scalar-comparator ${scalar_cmp} --substitute-cost-function ${subst_cost_func} --substitute-cost-modifier ${subst_cost_mod} --delete-cost-function ${delete_cost_func} --delete-cost-modifier ${delete_cost_mod} --wordspot-feature-weight 0.8 --wordspot-location-weight ${location_weight} ${RP_EXTRA_OPTIONS})
		    
# 		    set(command_opts_filename ${feature}--desc-${desc_size}--serial-${serial_cmp}--scalar-${scalar_cmp}--subst-func-${subst_cost_func}--subst-mod-${subst_cost_mod}--del-func-${delete_cost_func}--del-mod-${delete_cost_mod}--featw-0.8--locw-${location_weight}-${extra_opts_filename}--${outfile_suffix})
		    
# 		    set(output_file ${RP_OUTPUT_DIR}/${command_opts_filename}.csv)
# 		    set(log_file ${RP_OUTPUT_DIR}/${command_opts_filename}.log)
# 		    set(info_file ${RP_OUTPUT_DIR}/${command_opts_filename}.info)

# 		    if(${RUN_MEMCHECK})

# 		    add_custom_command(TARGET retrieval-recall-precision 
# 		      POST_BUILD
# 		      COMMAND echo Memcheck: ${output_file})

# 		      add_custom_command(TARGET retrieval-recall-precision
# 			POST_BUILD
# 			COMMAND valgrind --tool=memcheck --leak-check=yes ./retrieval-recall-precision --input ${RP_INPUT_DIR} --output-file ${output_file} ${command_opts} > ${log_file})


# 		    else() 
# 		      if (${RUN_PROFILER})

# 			add_custom_command(TARGET retrieval-recall-precision 
# 			  POST_BUILD
# 			  COMMAND echo Profiling: ${output_file})
			
# 			add_custom_command(TARGET retrieval-recall-precision
# 			  POST_BUILD
# 			  COMMAND perf record ./retrieval-recall-precision --input ${RP_INPUT_DIR} --output-file ${output_file} ${command_opts} > ${log_file})

# 			add_custom_command(TARGET retrieval-recall-precision
# 			  POST_BUILD
# 			  COMMAND xterm -e perf report)

# 		      else()

# 			add_custom_command(TARGET retrieval-recall-precision 
# 			  POST_BUILD
# 			  COMMAND echo Processing: ${output_file})
			
# 			add_custom_command(TARGET retrieval-recall-precision
# 			  POST_BUILD
# 			  COMMAND retrieval-recall-precision --input ${RP_INPUT_DIR} --output-file ${output_file} ${command_opts} > ${log_file})
			
# 			add_custom_command(TARGET retrieval-recall-precision
# 			  POST_BUILD
# 			  COMMAND head -n 100 ${log_file} > ${info_file})
			
# 			# add_custom_command(TARGET retrieval-recall-precision
# 			#   POST_BUILD
# 			#   COMMAND rm ${log_file})
			
# 			# add_custom_command(TARGET retrieval-recall-precision
# 			#   POST_BUILD
# 			#   COMMAND if [ -n `ls | grep "${output_file}.*csv$"` ] ; then ; mv ${output_file}*.csv ${RP_OUTPUT_DIR}/words/ ; fi )

# 		      endif()
# 		    endif()
# 		  endforeach(location_weight)
# 		endforeach(delete_cost_mod)
# 	      endforeach(delete_cost_func)
# 	    endforeach(subst_cost_mod)
# 	  endforeach(subst_cost_func)
# 	endforeach(desc_size)
#       endforeach(scalar_cmp)
#     endforeach(serial_cmp)
#   endforeach(feature)
# endmacro(run_recall_precision)

# function(recall_precision_test DATASET)
#   message(DATASET: ${DATASET})
#   set (RP_INPUT_DIR ${DATASET_DIR}/${DATASET})
#   set (RP_OUTPUT_DIR ${RESULT_DIR}/recall-precision/${DATASET})
#   set (RP_FEATURES ${ALL_SERIAL_FEATURES} )
#   set (RP_SERIAL_COMPARATORS dynamic-time-warping-normalized dynamic-minimums )
#   set (RP_SCALAR_COMPARATORS plain )
#   set (RP_DESC_SIZE 4)
#   set (RP_SUBST_FUNC const)
#   set (RP_SUBST_MOD 1)
#   set (RP_DELETE_FUNC pow)
#   set (RP_DELETE_MOD 2)
#   set (RP_LOCATION_WEIGHTS 1.6)
#   set (RP_EXTRA_OPTIONS --wordspot-use-feature-pool --use-multithreading) 
#   run_recall_precision()
# endfunction(recall_precision_test)

# ## RECALL PRECISION TESTS

# # recall_precision_test(${DATASET_DEBUG_TEST})
# # recall_precision_test(${DATASET_LM_5_I300})
# # recall_precision_test(${DATASET_LM_38})
# # recall_precision_test(${DATASET_LM_1229})
# # recall_precision_test(${DATASET_IAM_GW_16_I928})
# # recall_precision_test(${DATASET_IAM_GW_16_I4586})
# # recall_precision_test(${DATASET_IAM_GW_77})
# # recall_precision_test(${DATASET_IAM_GW_02})
# # recall_precision_test(${DATASET_IAM_GW_1472})
# # recall_precision_test(${DATASET_IFNENIT_5_I300})
# # recall_precision_test(${DATASET_IFNENIT_44_I1000})
# # recall_precision_test(${DATASET_IFNENIT_822})
# # recall_precision_test(${DATASET_KAGGLE_GENDER_HW_COMP})
# recall_precision_test(${DATASET_KUFIC_COMPONENTS})

# macro(run_similar_files)

#   add_custom_command(TARGET similar-files
#     POST_BUILD
#     COMMAND mkdir -p ${SF_OUTPUT_DIR}/words/)

#   string(REGEX REPLACE " +" "-" extra_opts_filename ${SF_EXTRA_OPTIONS})



#   foreach(feature ${SF_FEATURES})
#     foreach(serial_cmp ${SF_SERIAL_COMPARATORS})
#       foreach(scalar_cmp ${SF_SCALAR_COMPARATORS})
# 	foreach(desc_size ${SF_DESC_SIZE})
# 	  foreach(subst_cost_func ${SF_SUBST_FUNC})
# 	    foreach(subst_cost_mod ${SF_SUBST_MOD})
# 	      foreach(delete_cost_func ${SF_DELETE_FUNC})
# 		foreach(delete_cost_mod ${SF_DELETE_MOD})
# 		  foreach(location_weight ${SF_LOCATION_WEIGHTS})

# 		      string(RANDOM outfile_suffix)
              
# 		      set(command_opts -q ${SF_QUERY_DIR} --result-size 0 --feature ${feature} --descriptor-size ${desc_size} --serial-comparator ${serial_cmp} --scalar-comparator ${scalar_cmp} --substitute-cost-function ${subst_cost_func} --substitute-cost-modifier ${subst_cost_mod} --delete-cost-function ${delete_cost_func} --delete-cost-modifier ${delete_cost_mod} --wordspot-feature-weight 1.0 --wordspot-location-weight ${location_weight} ${SF_EXTRA_OPTIONS})
		    
# 		      set(command_opts_filename ${feature}--d-${desc_size}--ser-${serial_cmp}--sca-${scalar_cmp}--subst-${subst_cost_func}-${subst_cost_mod}--del-${delete_cost_func}-${delete_cost_mod}--featw-1.0--locw-${location_weight}-${extra_opts_filename}--${outfile_suffix})
		    
# 		    set(output_file ${SF_OUTPUT_DIR}/${command_opts_filename})
# 		    set(log_file ${SF_OUTPUT_DIR}/${command_opts_filename}.log)
# 		    set(info_file ${SF_OUTPUT_DIR}/${command_opts_filename}.info)

# 		    if(${RUN_MEMCHECK})

# 		    add_custom_command(TARGET similar-files 
# 		      POST_BUILD
# 		      COMMAND echo Memcheck: ${output_file})

# 		      add_custom_command(TARGET similar-files
# 			POST_BUILD
# 			COMMAND valgrind --tool=memcheck --leak-check=yes ./similar-files --input ${SF_INPUT_DIR} --output-file ${output_file} ${command_opts} > ${log_file})


# 		    else() 
# 		      if (${RUN_PROFILER})

# 			add_custom_command(TARGET similar-files 
# 			  POST_BUILD
# 			  COMMAND echo Profiling: ${output_file})
			
# 			add_custom_command(TARGET similar-files
# 			  POST_BUILD
# 			  COMMAND perf record ./similar-files --input ${SF_INPUT_DIR} --output-file ${output_file} ${command_opts} > ${log_file})

# 			add_custom_command(TARGET similar-files
# 			  POST_BUILD
# 			  COMMAND xterm -e perf report)

# 		      else()

# 			add_custom_command(TARGET similar-files 
# 			  POST_BUILD
# 			  COMMAND echo Processing: ${output_file})
			
# 			add_custom_command(TARGET similar-files
# 			  POST_BUILD
# 			  COMMAND similar-files --input ${SF_INPUT_DIR} --output-file ${output_file} ${command_opts} > ${log_file})
			
# 			add_custom_command(TARGET similar-files
# 			  POST_BUILD
# 			  COMMAND head -n 100 ${log_file} > ${info_file})
			
# 			# add_custom_command(TARGET similar-files
# 			#   POST_BUILD
# 			#   COMMAND rm ${log_file})
			
# 			# add_custom_command(TARGET similar-files
# 			#   POST_BUILD
# 			#   COMMAND if [ -n `ls | grep "${output_file}.*csv$"` ] ; then ; mv ${output_file}*.csv ${SF_OUTPUT_DIR}/words/ ; fi )

# 		      endif()
# 		    endif()
# 		  endforeach(location_weight)
# 		endforeach(delete_cost_mod)
# 	      endforeach(delete_cost_func)
# 	    endforeach(subst_cost_mod)
# 	  endforeach(subst_cost_func)
# 	endforeach(desc_size)
#       endforeach(scalar_cmp)
#     endforeach(serial_cmp)
#   endforeach(feature)
# endmacro(run_similar_files)

# function(similar_files_test DATASET)
#   message(DATASET: ${DATASET})
#   set (SF_INPUT_DIR ${DATASET_DIR}/${DATASET})
#   set (SF_QUERY_DIR ${DATASET_DIR}/${DATASET}-query)
#   set (SF_OUTPUT_DIR ${RESULT_DIR}/similar-files/${DATASET})
#   set (SF_FEATURES projection-profile contour-single-line sift shape-context )
#   set (SF_SERIAL_COMPARATORS dynamic-time-warping-normalized )
#   set (SF_SCALAR_COMPARATORS euclidean )
#   set (SF_DESC_SIZE 4)
#   set (SF_SUBST_FUNC const)
#   set (SF_SUBST_MOD 1)
#   set (SF_DELETE_FUNC pow)
#   set (SF_DELETE_MOD 2)
#   set (SF_LOCATION_WEIGHTS 0)
#   set (SF_EXTRA_OPTIONS --wordspot-use-feature-pool --use-multithreading ) 
#   run_similar_files()
# endfunction(similar_files_test)

# ## SIMILAR FILES

# # similar_files_test(${DATASET_DEBUG_TEST})
# # similar_files_test(${DATASET_LM_5_I300})
# # similar_files_test(${DATASET_LM_38})
# # similar_files_test(${DATASET_LM_1229})
# # similar_files_test(${DATASET_IAM_GW_16_I928})
# # similar_files_test(${DATASET_IAM_GW_16_I4586})
# # similar_files_test(${DATASET_IAM_GW_77})
# # similar_files_test(${DATASET_IAM_GW_02})
# # similar_files_test(${DATASET_IAM_GW_1472})
# # similar_files_test(${DATASET_IFNENIT_5_I300})
# # similar_files_test(${DATASET_IFNENIT_44_I1000})
# # similar_files_test(${DATASET_IFNENIT_822})
# # similar_files_test(${DATASET_KAGGLE_GENDER_HW_COMP})
# # similar_files_test(${DATASET_KUFIC_COMPONENTS})

# macro(run_similar_matches)

#   add_custom_command(TARGET similar-matches
#     POST_BUILD
#     COMMAND mkdir -p ${SM_OUTPUT_DIR}/words/)

#   string(REGEX REPLACE " +" "-" extra_opts_filename ${SM_EXTRA_OPTIONS})

#   foreach(feature ${SM_FEATURES})
#     foreach(serial_cmp ${SM_SERIAL_COMPARATORS})
#       foreach(scalar_cmp ${SM_SCALAR_COMPARATORS})
# 	    foreach(desc_size ${SM_DESC_SIZE})
# 	      foreach(subst_cost_func ${SM_SUBST_FUNC})
# 	        foreach(subst_cost_mod ${SM_SUBST_MOD})
# 	          foreach(delete_cost_func ${SM_DELETE_FUNC})
# 		        foreach(delete_cost_mod ${SM_DELETE_MOD})
# 		          foreach(location_weight ${SM_LOCATION_WEIGHTS})

# 		            string(RANDOM outfile_suffix)
                    
# 		            set(command_opts -q ${SM_QUERY_FILE} --result-size 0 --feature ${feature} --descriptor-size ${desc_size} --serial-comparator ${serial_cmp} --scalar-comparator ${scalar_cmp} --substitute-cost-function ${subst_cost_func} --substitute-cost-modifier ${subst_cost_mod} --delete-cost-function ${delete_cost_func} --delete-cost-modifier ${delete_cost_mod} --wordspot-feature-weight 1.0 --wordspot-location-weight ${location_weight} ${SM_EXTRA_OPTIONS})
		    
# 		            set(command_opts_filename ${feature}--d-${desc_size}--ser-${serial_cmp}--sca-${scalar_cmp}--subst-${subst_cost_func}-${subst_cost_mod}--del-${delete_cost_func}-${delete_cost_mod}--featw-1.0--locw-${location_weight}-${extra_opts_filename}--${outfile_suffix})
		    
# 		            set(output_file ${SM_OUTPUT_DIR}/${command_opts_filename})
# 		            set(log_file ${SM_OUTPUT_DIR}/${command_opts_filename}.log)
# 		            set(info_file ${SM_OUTPUT_DIR}/${command_opts_filename}.info)
                    
# 		            if(${RUN_MEMCHECK})
                      
# 		              add_custom_command(TARGET similar-matches 
# 		                POST_BUILD
# 		                COMMAND echo Memcheck: ${output_file})
                      
# 		              add_custom_command(TARGET similar-matches
# 			            POST_BUILD
# 			            COMMAND valgrind --tool=memcheck --leak-check=yes ./similar-matches --input ${SM_INPUT_DIR} --output-file ${output_file} ${command_opts} > ${log_file})
                      
                      
# 		            else() 
# 		              if (${RUN_PROFILER})
                        
# 			            add_custom_command(TARGET similar-matches 
# 			              POST_BUILD
# 			              COMMAND echo Profiling: ${output_file})
			            
# 			            add_custom_command(TARGET similar-matches
# 			              POST_BUILD
# 			              COMMAND perf record ./similar-matches --input ${SM_INPUT_DIR} --output-file ${output_file} ${command_opts} > ${log_file})
                        
# 			            add_custom_command(TARGET similar-matches
# 			              POST_BUILD
# 			              COMMAND xterm -e perf report)
                        
# 		              else()
                        
# 			            add_custom_command(TARGET similar-matches 
# 			              POST_BUILD
# 			              COMMAND echo Processing: ${output_file})
			            
# 			            add_custom_command(TARGET similar-matches
# 			              POST_BUILD
# 			              COMMAND similar-matches --input ${SM_LIBRARY_DIR} --output-file ${output_file} ${command_opts} > ${log_file})
			            
# 			            add_custom_command(TARGET similar-matches
# 			              POST_BUILD
# 			              COMMAND head -n 100 ${log_file} > ${info_file})
			
# 			            # add_custom_command(TARGET similar-matches
# 			            #   POST_BUILD
# 			            #   COMMAND rm ${log_file})
			            
# 			            # add_custom_command(TARGET similar-matches
# 			            #   POST_BUILD
# 			            #   COMMAND if [ -n `ls | grep "${output_file}.*csv$"` ] ; then ; mv ${output_file}*.csv ${SM_OUTPUT_DIR}/words/ ; fi )
                        
# 		              endif()
# 		            endif()
# 		          endforeach(location_weight)
# 		        endforeach(delete_cost_mod)
# 	          endforeach(delete_cost_func)
# 	        endforeach(subst_cost_mod)
# 	      endforeach(subst_cost_func)
# 	    endforeach(desc_size)
#       endforeach(scalar_cmp)
#     endforeach(serial_cmp)
#   endforeach(feature)
# endmacro(run_similar_matches)

# function(similar_matches_test CC_LIB QUERY_DIR)
#   message(DATASET: ${DATASET})
#   set (SM_LIBRARY_DIR ${DATASET_DIR}/${CC_LIB})
#   set (SM_QUERY_FILE ${DATASET_DIR}/${QUERY_DIR}/1.png)
#   set (SM_OUTPUT_DIR ${RESULT_DIR}/similar-matches/${QUERY_DIR})
#   set (SM_FEATURES chamfer )
#   set (SM_SERIAL_COMPARATORS dynamic-time-warping-normalized )
#   set (SM_SCALAR_COMPARATORS plain )
#   set (SM_DESC_SIZE 4)
#   set (SM_SUBST_FUNC const)
#   set (SM_SUBST_MOD 1)
#   set (SM_DELETE_FUNC pow)
#   set (SM_DELETE_MOD 2)
#   set (SM_LOCATION_WEIGHTS 0)
#   set (SM_EXTRA_OPTIONS " ") 
#   run_similar_matches()
# endfunction(similar_matches_test)

# ## SIMILAR FILES

# similar_matches_test(${DATASET_LM_COMPONENT_LIB} ${DATASET_LM_PAGES})

# ## OTAP_CALC_DIST

# macro(run_otap_calc_dist)

#   add_custom_command(TARGET otap-calc-dist
#     POST_BUILD
#     COMMAND mkdir -p ${OCD_OUTPUT_DIR})

#   string(REGEX REPLACE " +" "-" extra_opts_filename ${OCD_EXTRA_OPTIONS})
#   foreach(comparison_method ${OCD_COMPARISON_METHODS})
# 	string(RANDOM outfile_suffix)
# 	set(command_opts -i ${OCD_INPUT_FILE} -q ${OCD_QUERY_FILE} --method ${comparison_method} ${OCD_EXTRA_OPTIONS})
#     get_filename_component(input_we ${OCD_INPUT_FILE} NAME_WE)
#     get_filename_component(query_we ${OCD_QUERY_FILE} NAME_WE)
# 	set(command_opts_filename ${input_we}--query-${query_we}--method-${comparison_method}--${extra_opts_filename}--${outfile_suffix})
# 	set(output_file ${OCD_OUTPUT_DIR}/${command_opts_filename}.csv)
# 	set(log_file ${OCD_OUTPUT_DIR}/${command_opts_filename}.log)
# 	set(info_file ${OCD_OUTPUT_DIR}/${command_opts_filename}.info)
    
# 	if(${RUN_MEMCHECK})
      
# 	  add_custom_command(TARGET otap-calc-dist 
# 		POST_BUILD
# 		COMMAND echo Memcheck: ${output_file})
      
# 	  add_custom_command(TARGET otap-calc-dist
# 		POST_BUILD
# 		COMMAND valgrind --tool=memcheck --leak-check=yes ./otap-calc-dist --output-file ${output_file} ${command_opts} > ${log_file})
      
                      
# 	else() 
# 	  if (${RUN_PROFILER})
        
# 		add_custom_command(TARGET otap-calc-dist 
# 		  POST_BUILD
# 		  COMMAND echo Profiling: ${output_file})
		
# 		add_custom_command(TARGET otap-calc-dist
# 		  POST_BUILD
# 		  COMMAND perf record ./otap-calc-dist --output-file ${output_file} ${command_opts} > ${log_file})
        
# 		add_custom_command(TARGET otap-calc-dist
# 		  POST_BUILD
# 		  COMMAND xterm -e perf report)
        
# 	  else()
        
# 		add_custom_command(TARGET otap-calc-dist 
# 		  POST_BUILD
# 		  COMMAND echo Processing: ${output_file})
		
# 		add_custom_command(TARGET otap-calc-dist
# 		  POST_BUILD
# 		  COMMAND otap-calc-dist --output-file ${output_file} ${command_opts} > ${log_file})
		
# 		add_custom_command(TARGET otap-calc-dist
# 		  POST_BUILD
# 		  COMMAND head -n 100 ${log_file} > ${info_file})
		
# 		# add_custom_command(TARGET otap-calc-dist
# 		#   POST_BUILD
# 		#   COMMAND rm ${log_file})
		
# 		# add_custom_command(TARGET otap-calc-dist
# 		#   POST_BUILD
# 		#   COMMAND if [ -n `ls | grep "${output_file}.*csv$"` ] ; then ; mv ${output_file}*.csv ${OCD_OUTPUT_DIR}/words/ ; fi )
        
# 	  endif()
# 	endif()
#   endforeach(comparison_method)
# endmacro(run_otap_calc_dist)

# function(otap_calc_dist_test INPUT_FILE QUERY_FILE)
#   message(DATASET: ${DATASET})
#   set (OCD_COMPARISON_METHODS dtw emd)
#   ##  set (OCD_QUERY_FILE ${DATASET_DIR}/${QUERY_DIR}/1.png)
#   set (OCD_QUERY_FILE ${QUERY_FILE})
#   set (OCD_INPUT_FILE ${INPUT_FILE})
#   set (OCD_OUTPUT_DIR ${RESULT_DIR}/otap-calc-dist)
# ##  set (OCD_OUTPUT_DIR "")
#   set (OCD_EXTRA_OPTIONS " ") 
#   run_otap_calc_dist()
# endfunction(otap_calc_dist_test)

## CALC FILES

# otap_calc_dist_test("/home/iesahin/Repository/emresahin/work/12457-54854/332_05_block_pixel_histogram.csv" "/home/iesahin/Repository/emresahin/work/12457-54854/query_block_pixel_histogram.csv")
