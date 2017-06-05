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

VectorXd Tools::ConvertPolarToCartesian(const VectorXd& x)
{
  float rho = x(0);
  float phi = x(1);
  float rho_dot = x(2);

  float px = rho * std::cos(phi);
  float py = rho * std::sin(phi);
  float vel_abs = 0;
  float yaw_angle = 0;
  float yaw_rate = 0;

  VectorXd cartesian(5, 1);
  cartesian << px,
               py,
               vel_abs,
               yaw_angle,
               yaw_rate;

  return cartesian;
}
