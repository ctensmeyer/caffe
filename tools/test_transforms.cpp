
#include <cstring>
#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <stdio.h>
#include <opencv2/opencv.hpp>

#include "caffe/caffe.hpp"
#include "caffe/util/io.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"

using std::ofstream;

using namespace caffe;  // NOLINT(build/namespaces)

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  if (argc != 6) {
    LOG(ERROR) << "Usage: "
        << "test_transforms transform_prototxt_file doc_datum_lmdb num_images num_transforms output_folder";
    return 1;
  }

  // construct image transformer
  ImageTransformationParameter transform_param;
  string input_filename(argv[1]);
  if (!ReadProtoFromTextFile(input_filename, &transform_param)) {
    LOG(ERROR) << "Failed to parse input text file as ImageTransformationParameter: "
               << input_filename;
    return 2;
  }
  ImageTransformer<float>* transformer = CreateImageTransformer<float>(transform_param);

  // open LMDB for reading
  shared_ptr<db::DB> db;
  shared_ptr<db::Cursor> cursor;
  db.reset(db::GetDB("lmdb"));
  db->Open(argv[2], db::READ);
  cursor.reset(db->NewCursor());

  // start pulling entries from LMDB
  int num_images = atoi(argv[3]);
  int num_transforms = atoi(argv[4]);
  string out_dir(argv[5]);
  for (int i = 0; i < num_images; i++) {
    DocumentDatum doc;
    doc.ParseFromString(cursor->value());
    vector<int> in_shape;
    in_shape.push_back(1);
    in_shape.push_back(doc.image().channels());
    in_shape.push_back(doc.image().width());
    in_shape.push_back(doc.image().height());
    transformer->SampleTransformParams(in_shape);
	cv::Mat pretransform_img = ImageToCVMat(doc.image(), doc.image().channels() == 3);
	for (int j = 0; j < num_transforms; j++) {
	  string out_file = out_dir + "/" + std::to_string(i) + "_" + std::to_string(j) + ".png";
	  cv::Mat posttransform_img;
	  transformer->Transform(pretransform_img, posttransform_img);
	  cv::imwrite(out_file.c_str(), posttransform_img);
	}
	string out_file = out_dir + "/" + std::to_string(i) + ".png";
	cv::imwrite(out_file.c_str(), pretransform_img);
	cursor->Next();
  }

  delete transformer;
  return 0;
}
