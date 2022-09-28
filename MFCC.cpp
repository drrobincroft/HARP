#pragma once

#include "mfcc.cc"
#include "cnpy.h"

void CalMFCCForTraining(std::string wavpath, std::string mfccpath)
{
	cnpy::NpyArray arr = cnpy::npy_load(wavpath);
	float* d = arr.data<float>();
	short* d2 = new short[arr.shape[1]];

	size_t i, j, p, q;
	MFCC* mfcc = new MFCC(16000, 26, 25, 10, 26, 0.0, 16000 / 2.0, true);

	for (j = 0; j < arr.shape[1]; j++)
		d2[j] = (short)((int)(d[j] * 32767.0f + 32768.5f) - 32768);
	std::vector<std::vector<double>>& features = mfcc->process(d2, arr.shape[1]);

	int size3[] = { arr.shape[0],features.at(0).size(),features.size() };
	cv::Mat s(3, size3, CV_32FC1);
	for (i = 0; i < arr.shape[0]; i++)
	{
		for (j = 0; j < arr.shape[1]; j++)
			d2[j] = (short)((int)(d[i * arr.shape[1] + j] * 32767.0f + 32768.5f) - 32768);

		std::vector<std::vector<double>>& features = mfcc->process(d2, arr.shape[1]);
		for (p = 0; p < features.size(); p++)
		{
			for (q = 0; q < features.at(p).size(); q++)
			{
				s.at<float>(i, q, p) = (float)features.at(p).at(q);
			}
		}
		cout << (i+1) << "//" << arr.shape[0] << endl;
	}
	cnpy::npy_save<float>(mfccpath, (float*)s.data, { (size_t)size3[0],(size_t)size3[1],(size_t)size3[2] });

	delete mfcc;
	delete[] d2;
}

void main(void)
{
  //This is just an example
  CalMFCCForTraining("../Gesture/ellen/train_noduplication/wav.npy","../Gesture/ellen/train_noduplication/mfcc.npy");
}
