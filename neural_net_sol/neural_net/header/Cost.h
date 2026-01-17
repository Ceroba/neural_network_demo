#pragma once
#include<array>
#include<vector>
#include<cmath>
#include<iostream>
#include <random>
namespace cst {
	constexpr double EPS = 1e-12;

	// Cross entropy loss for one sample
	double cross_entropy(const std::vector<double>& expect, const std::vector<double>& pred) {
		double c = 0.0;
		for (int i = 0; i < expect.size(); i++) {
			c += expect[i] * std::log(pred[i] + EPS);
		}
		return -c;
	}
	double MSE(const std::vector<double>& expect, const std::vector<double>& pred) {
		double c = 0;
		for (int i = 0; i < expect.size(); i++) {
			double diff = expect[i] - pred[i];
			c += diff * diff;
		}
		return c / expect.size();
	}
	double d_MSE(double expect, double pred) {
		return pred - expect;
	}
}