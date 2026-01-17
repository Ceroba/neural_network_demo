#pragma once
#include<array>
#include<vector>
#include<cmath>
#include<iostream>
#include <random>
namespace act{
	enum act_type
	{
		sigmoid = 0,
		relu = 1,
		tanh = 2,
		softmax = 3
	};
		inline double Sigmoid(double x) {
			return 1.0 / (1.0 + std::exp(-x));
		}
		inline double D_Sigmoid(double x) {
			double act = Sigmoid(x);
			return act * (1 - act);
		}
		inline double ReLu(double x) {
			return std::max(0.0, x);
		}
		inline double D_ReLu(double x) {
			return x > 0.0 ? 1.0 : 0.0;
		}
		inline double TanH(double x) {
			return std::tanh(x);
		}
		inline double D_TanH(double x) {
			double t = TanH(x);
			return 1 - t * t;
		}
		inline std::vector<double> Softmax(const std::vector<double>& x) {
			std::vector<double> out(x.size());
			double max_x = *std::max_element(x.begin(), x.end()); // for numerical stability
			double sum = 0.0;

			// Calculate exponentials (shifted by max_x)
			for (double v : x)
				sum += std::exp(v - max_x);

			// Normalize to probabilities
			for (size_t i = 0; i < x.size(); i++)
				out[i] = std::exp(x[i] - max_x) / sum;

			return out;
		}
}