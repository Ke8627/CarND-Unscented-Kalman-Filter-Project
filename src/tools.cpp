#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth)
{
  VectorXd rmse(4);

  rmse << 0,0,0,0;

  // Check the validity of the inputs.
  if (estimations.size() != ground_truth.size() || estimations.size() == 0)
  {
    throw std::runtime_error("Invalid estimation or ground_truth data");
  }

  // Accumulate squared residuals.
  for (unsigned int i=0; i < estimations.size(); ++i)
  {
    VectorXd residual = estimations[i] - ground_truth[i];

    // Coefficient-wise multiplication.
    residual = residual.array()*residual.array();

    rmse += residual;
  }

  // Calculate the mean.
  rmse = rmse/estimations.size();

  rmse = rmse.array().sqrt();

  return rmse;
}
