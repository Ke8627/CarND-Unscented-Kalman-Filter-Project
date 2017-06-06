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
  // If this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // If this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  n_x_ = 5;

  n_aug_ = n_x_ + 2;

  // Spreading parameter
  lambda_ = 3 - n_x_;

  // Initial state vector
  x_ = VectorXd(n_x_);

  // TODO: Improve P_ initialization.
  // Initial covariance matrix
  P_ = MatrixXd::Identity(n_x_, n_x_);

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

  // Create vector for weights.
  weights_ = VectorXd(2 * n_aug_ + 1);

  // Set weights.
  double weight_0 = lambda_ / (lambda_ + n_aug_);
  weights_(0) = weight_0;
  for (int i = 1; i < 2 * n_aug_ + 1; i++)
  {
    double weight = 0.5 / (n_aug_ + lambda_);
    weights_(i) = weight;
  }

  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(const MeasurementPackage& measurement)
{
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

  auto Xsig_pred = Prediction(delta_t);

  // Switch between radar and lidar measurements.
  if (measurement.sensor_type_ == MeasurementPackage::RADAR)
  {
    UpdateRadar(measurement, Xsig_pred);
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
MatrixXd UKF::Prediction(double delta_t)
{
  // Estimate the object's location.

  auto Xsig_pred = PredictSigmaPoints(delta_t);

  PredictMeanAndCovariance(Xsig_pred);

  return Xsig_pred;
}

MatrixXd UKF::PredictSigmaPoints(double delta_t)
{
  auto Xsig_aug = GenerateAugmentedSigmaPoints();

  auto Xsig_pred = MatrixXd(n_x_, 2 * n_aug_ + 1);

  // Predict sigma points.
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    // Extract values for better readability.
    double p_x = Xsig_aug(0, i);
    double p_y = Xsig_aug(1, i);
    double v = Xsig_aug(2, i);
    double yaw = Xsig_aug(3, i);
    double yawd = Xsig_aug(4, i);
    double nu_a = Xsig_aug(5, i);
    double nu_yawdd = Xsig_aug(6, i);

    // Predicted state values.
    double px_p, py_p;

    // Avoid division by zero.
    if (fabs(yawd) > 0.001)
    {
        px_p = p_x + v / yawd * (sin(yaw + yawd * delta_t) - sin(yaw));
        py_p = p_y + v / yawd * (cos(yaw) - cos(yaw + yawd * delta_t));
    }
    else
    {
        px_p = p_x + v * delta_t * cos(yaw);
        py_p = p_y + v * delta_t * sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd * delta_t;
    double yawd_p = yawd;

    // Add noise.
    px_p = px_p + 0.5 * nu_a * delta_t * delta_t * cos(yaw);
    py_p = py_p + 0.5 * nu_a * delta_t * delta_t * sin(yaw);
    v_p = v_p + nu_a * delta_t;

    yaw_p = yaw_p + 0.5 * nu_yawdd * delta_t * delta_t;
    yawd_p = yawd_p + nu_yawdd * delta_t;

    // Write predicted sigma point into proper column.
    Xsig_pred(0, i) = px_p;
    Xsig_pred(1, i) = py_p;
    Xsig_pred(2, i) = v_p;
    Xsig_pred(3, i) = yaw_p;
    Xsig_pred(4, i) = yawd_p;
  }

  return Xsig_pred;
}

static void NormalizeAngle(double& angle)
{
  static const double two_pi = 2 * M_PI;

  while (angle > M_PI)
  {
    angle -= two_pi;
  }
  while (angle < -M_PI)
  {
    angle += two_pi;
  }
}

void UKF::PredictMeanAndCovariance(const MatrixXd& Xsig_pred)
{
  // Predicted state mean
  x_.fill(0.0);
  // Iterate over sigma points.
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    x_ = x_ + weights_(i) * Xsig_pred.col(i);
  }

  // Predicted state covariance matrix
  P_.fill(0.0);
  // Iterate over sigma points.
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    // Calculate state difference.
    VectorXd x_diff = Xsig_pred.col(i) - x_;
    NormalizeAngle(x_diff(3));

    P_ = P_ + weights_(i) * x_diff * x_diff.transpose();
  }
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

MatrixXd UKF::GenerateAugmentedSigmaPoints()
{
  // Create augmented mean vector.
  VectorXd x_aug = VectorXd(n_aug_);

  // Create augmented state covariance.
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);

  // Create sigma point matrix.
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  // Create augmented mean state.
  x_aug.head(n_x_) = x_;
  x_aug(n_x_) = 0;
  x_aug(n_aug_ + 1) = 0;

  // Create augmented covariance matrix.
  P_aug.fill(0.0);
  P_aug.topLeftCorner(n_x_,n_x_) = P_;
  P_aug(n_x_ , n_x_) = std_a_ * std_a_;
  P_aug(n_aug_ + 1, n_aug_ + 1) = std_yawdd_ * std_yawdd_;

  // Create square root matrix.
  MatrixXd L = P_aug.llt().matrixL();

  // Create augmented sigma points.
  Xsig_aug.col(0) = x_aug;
  for (int i = 0; i < n_aug_; i++)
  {
    Xsig_aug.col(i + 1)          = x_aug + sqrt(lambda_ + n_aug_) * L.col(i);
    Xsig_aug.col(i + 1 + n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * L.col(i);
  }

  return Xsig_aug;
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

static const int c_radarMeasurementSize = 3;

MatrixXd UKF::TransformSigmaPointsToRadarSpace(const MatrixXd& Xsig_pred)
{
  MatrixXd Zsig = MatrixXd(c_radarMeasurementSize, 2 * n_aug_ + 1);

  // Transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    // Extract values for better readability.
    double p_x = Xsig_pred(0, i);
    double p_y = Xsig_pred(1, i);
    double v  = Xsig_pred(2, i);
    double yaw = Xsig_pred(3, i);

    double v1 = cos(yaw) * v;
    double v2 = sin(yaw) * v;

    // Measurement model
    // r
    Zsig(0, i) = sqrt(p_x * p_x + p_y * p_y);
    // phi
    Zsig(1, i) = atan2(p_y, p_x);
    // r_dot
    Zsig(2, i) = (p_x * v1 + p_y * v2 ) / sqrt(p_x * p_x + p_y * p_y);
  }

  return Zsig;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(const MeasurementPackage& measurement, const MatrixXd& Xsig_pred)
{
  // Use radar data to update the belief about the object's position.
  // Modify the state vector, x_, and covariance, P_.

  // TODO: Calculate the radar NIS.

  static const int n_z = c_radarMeasurementSize;

  auto Zsig = TransformSigmaPointsToRadarSpace(Xsig_pred);

  // Mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  // Measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z, n_z);
  S.fill(0.0);
  // Iterate through sigma points.
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    // Residual.
    VectorXd z_diff = Zsig.col(i) - z_pred;

    NormalizeAngle(z_diff(1));

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  // Add measurement noise covariance matrix.
  MatrixXd R = MatrixXd(n_z, n_z);
  R <<    std_radr_ * std_radr_, 0, 0,
          0, std_radphi_ * std_radphi_, 0,
          0, 0, std_radrd_ * std_radrd_;
  S = S + R;

  // Create matrix for cross correlation Tc.
  MatrixXd Tc = MatrixXd(n_x_, n_z);

  // Calculate cross correlation matrix.
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    // Residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    NormalizeAngle(z_diff(1));

    // State difference
    VectorXd x_diff = Xsig_pred.col(i) - x_;

    NormalizeAngle(x_diff(3));

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  // Kalman gain K
  MatrixXd K = Tc * S.inverse();

  // Residual
  VectorXd z_diff = measurement.raw_measurements_ - z_pred;

  NormalizeAngle(z_diff(1));

  // Update state mean and covariance matrix.
  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();
}
