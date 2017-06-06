#include "ukf.h"
#include "Eigen/Dense"
#include "tools.h"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  n_x_ = 5;

  // define spreading parameter
  lambda_ = 3 - n_x_;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 30;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 30;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */

  // TODO: Initialize P_.
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(const MeasurementPackage& measurement)
{
  /**
  TODO:

  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */
  if (!is_initialized_)
  {
    if (measurement.sensor_type_ == MeasurementPackage::RADAR)
    {
      x_ = Tools::ConvertPolarToCartesian(measurement.raw_measurements_);
    }
    else if (measurement.sensor_type_ == MeasurementPackage::LASER)
    {
      x_ << measurement.raw_measurements_(0),
            measurement.raw_measurements_[1],
            0,
            0,
            0;
    }

    is_initialized_ = true;
    time_us_ = measurement.timestamp_;
    return;
  }

  double delta_t = measurement.timestamp_ - time_us_;

  Prediction(delta_t);

  if (measurement.sensor_type_ == MeasurementPackage::RADAR)
  {
    UpdateRadar(measurement);
  }
  else if (measurement.sensor_type_ == MeasurementPackage::LASER)
  {
    UpdateLidar(measurement);
  }

  time_us_ = measurement.timestamp_;
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t)
{
  /**
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */

  auto XSig = GenerateSigmaPoints();
}

MatrixXd UKF::GenerateSigmaPoints()
{
  // Create sigma point matrix.
  MatrixXd Xsig = MatrixXd(n_x_, 2 * n_x_ + 1);

  // Calculate square root of P.
  MatrixXd A = P_.llt().matrixL();

  // Set first column of sigma point matrix.
  Xsig.col(0) = x_;

  // Set remaining sigma points.
  for (int i = 0; i < n_x_; i++)
  {
    Xsig.col(i + 1)        = x_ + sqrt(lambda_ + n_x_) * A.col(i);
    Xsig.col(i + 1 + n_x_) = x_ - sqrt(lambda_ + n_x_) * A.col(i);
  }

  return Xsig;
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(const MeasurementPackage& measurement)
{
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(const MeasurementPackage& measurement)
{
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */
}
