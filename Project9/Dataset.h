#pragma once
#include <vector>
#include <cmath>
class Dataset
{
	std::vector<double> m_x_train;
	std::vector<double> m_y_train;
	std::vector<double> m_x_val;
	std::vector<double> m_y_val;




public:
	Dataset();
	std::vector<double> getXTrain();
	std::vector<double> getYTrain();
	std::vector<double> getXVal();
	std::vector<double> getYVal();



};

