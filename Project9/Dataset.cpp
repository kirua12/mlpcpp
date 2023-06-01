#include "Dataset.h"

Dataset::Dataset()
{
	double Pi = std::acos(-1);
	
	int count = 0;
	for (double x = -2 * Pi; x <= 2 * Pi;x += 0.03) {
		if (count % 10 < 8) {
			m_x_train.push_back(x);
			double y = std::sin(2 * x);
			m_y_train.push_back(y);
		}
		else {
			m_x_val.push_back(x);
			double y = std::sin(2 * x);
			m_y_val.push_back(y);
		}


		count++;
	}

}

std::vector<double> Dataset::getXTrain()
{
	return m_x_train;
}

std::vector<double> Dataset::getYTrain()
{
	return m_y_train;
}

std::vector<double> Dataset::getXVal()
{
	return m_x_val;
}

std::vector<double> Dataset::getYVal()
{
	return m_y_val;
}



