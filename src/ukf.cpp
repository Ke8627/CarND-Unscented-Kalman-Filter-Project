#include "ukf.h"
#include "Eigen/Dense"
#include "tools.h"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

static const int c_radarMeasurementSize = 3;
static const int c_lidarMeasurementSize = 2;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF()
{
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

  // Initial covariance matrix
  P_ = MatrixXd::Identity(n_x_, n_x_) / 5;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 2;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.2;

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

  R_radar_ = MatrixXd(c_radarMeasurementSize, c_radarMeasurementSize);
  R_radar_ << std_radr_ * std_radr_, 0, 0,
              0, std_radphi_ * std_radphi_, 0,
              0, 0, std_radrd_ * std_radrd_;

  R_lidar_ = MatrixXd(c_lidarMeasurementSize, c_lidarMeasurementSize);
  R_lidar_ << std_laspx_ * std_laspx_, 0,
              0, std_laspy_ * std_laspy_;
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

  static const double c_microsecondsPerSecond = 1000000;

  double delta_t = (measurement.timestamp_ - time_us_) / c_microsecondsPerSecond;

  const auto Xsig_pred = Prediction(delta_t);

  // Switch between radar and lidar measurements.
  if (measurement.sensor_type_ == MeasurementPackage::RADAR)
  {
    if (use_radar_)
    {
      UpdateRadar(measurement, Xsig_pred);

      time_us_ = measurement.timestamp_;
    }
  }
  else if (measurement.sensor_type_ == MeasurementPackage::LASER)
  {
    if (use_laser_)
    {
      UpdateLidar(measurement, Xsig_pred);

      time_us_ = measurement.timestamp_;
    }
  }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
MatrixXd UKF::Prediction(double delta_t)
{
  // Estimate the object's location.

  const auto Xsig_pred = PredictSigmaPoints(delta_t);

  PredictMeanAndCovariance(Xsig_pred);

  return Xsig_pred;
}

MatrixXd UKF::PredictSigmaPoints(double delta_t)
{
  const auto Xsig_aug = GenerateAugmentedSigmaPoints();

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
  x_aug(n_x_ + 1) = 0;

  // Create augmented covariance matrix.
  P_aug.fill(0.0);
  P_aug.topLeftCorner(n_x_,n_x_) = P_;
  P_aug(n_x_ , n_x_) = std_a_ * std_a_;
  P_aug(n_x_ + 1, n_x_ + 1) = std_yawdd_ * std_yawdd_;

  // Create square root matrix.
  const MatrixXd L = P_aug.llt().matrixL();

  // Create augmented sigma points.
  Xsig_aug.col(0) = x_aug;
  for (int i = 0; i < n_aug_; i++)
  {
    Xsig_aug.col(i + 1)          = x_aug + sqrt(lambda_ + n_aug_) * L.col(i);
    Xsig_aug.col(i + 1 + n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * L.col(i);
  }

  return Xsig_aug;
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

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(const MeasurementPackage& measurement, const MatrixXd& Xsig_pred)
{
  // Use lidar data to update the belief about the object's position.
  // Modify the state vector, x_, and covariance, P_.

  // TODO: Calculate the lidar NIS.

  const int n_z = c_lidarMeasurementSize;

  const auto Zsig = TransformSigmaPointsToLidarSpace(Xsig_pred);

  const auto z_pred = GetMeanPredictedMeasurement(Zsig, n_z);

  const auto S = CalculateMeasurementCovariance(Zsig,
                                                z_pred,
                                                R_lidar_,
                                                n_z);

  const auto Tc = CalculateCrossCorrelation(Zsig,
                                            z_pred,
                                            Xsig_pred,
                                            n_z);

  UpdateFromMeasurement(Tc,
                        z_pred,
                        S,
                        measurement.raw_measurements_);
}

MatrixXd UKF::TransformSigmaPointsToLidarSpace(const MatrixXd& Xsig_pred)
{
  MatrixXd Zsig = MatrixXd(c_lidarMeasurementSize, 2 * n_aug_ + 1);

  // Transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    // Extract values for better readability.
    double p_x = Xsig_pred(0, i);
    double p_y = Xsig_pred(1, i);

    // Measurement model
    Zsig(0, i) = p_x;
    Zsig(1, i) = p_y;
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

  const auto Zsig = TransformSigmaPointsToRadarSpace(Xsig_pred);

  const auto z_pred = GetMeanPredictedMeasurement(Zsig, n_z);

  const int angle_index = 1;
  const auto S = CalculateMeasurementCovariance(Zsig,
                                                z_pred,
                                                R_radar_,
                                                n_z,
                                                &angle_index);

  const auto Tc = CalculateCrossCorrelation(Zsig,
                                            z_pred,
                                            Xsig_pred,
                                            n_z,
                                            &angle_index);

  UpdateFromMeasurement(Tc,
                        z_pred,
                        S,
                        measurement.raw_measurements_,
                        &angle_index);
}

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

    double c1 = sqrt(p_x * p_x + p_y * p_y);

    if (fabs(c1) < 0.0001)
    {
      c1 = 0.0001;
      std::cout << "Note: Avoided division by zero." << std::endl;
    }

    // Measurement model
    // r
    Zsig(0, i) = c1;
    // phi
    Zsig(1, i) = atan2(p_y, p_x);
    // r_dot
    Zsig(2, i) = (p_x * v1 + p_y * v2 ) / c1;
  }

  return Zsig;
}

VectorXd UKF::GetMeanPredictedMeasurement(const MatrixXd& Zsig, int n_z)
{
  // Mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  return z_pred;
}

MatrixXd UKF::CalculateMeasurementCovariance(const MatrixXd& Zsig,
                                             const VectorXd& z_pred,
                                             const MatrixXd& R,
                                             int n_z,
                                             const int* angle_index)
{
  // Measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z, n_z);
  S.fill(0.0);
  // Iterate through sigma points.
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    // Residual.
    VectorXd z_diff = Zsig.col(i) - z_pred;

    if (angle_index != nullptr)
    {
      NormalizeAngle(z_diff(*angle_index));
    }

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  // Add measurement noise covariance matrix.
  S = S + R;

  return S;
}

MatrixXd UKF::CalculateCrossCorrelation(const MatrixXd& Zsig,
                                        const VectorXd& z_pred,
                                        const MatrixXd& Xsig_pred,
                                        int n_z,
                                        const int* angle_index)
{
  // Create matrix for cross correlation Tc.
  MatrixXd Tc = MatrixXd(n_x_, n_z);

  // Calculate cross correlation matrix.
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    // Residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    if (angle_index != nullptr)
    {
      NormalizeAngle(z_diff(*angle_index));
    }

    // State difference
    VectorXd x_diff = Xsig_pred.col(i) - x_;

    NormalizeAngle(x_diff(3));

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  return Tc;
}

void UKF::UpdateFromMeasurement(const MatrixXd& Tc,
                                const VectorXd& z_pred,
                                const MatrixXd& S,
                                const VectorXd& raw_measurements,
                                const int* angle_index)
{
  // Kalman gain K
  const MatrixXd K = Tc * S.inverse();

  // Residual
  VectorXd z_diff = raw_measurements - z_pred;

  if (angle_index != nullptr)
  {
    NormalizeAngle(z_diff(*angle_index));
  }

  // Update state mean and covariance matrix.
  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();
}

void UKF::NormalizeAngle(double& angle)
{
  static const double two_pi = 2 * M_PI;

  // Shift from [-pi, pi) to [0, 2pi).
  angle += M_PI;

  // Normalize to [0, 2pi).
  double remainder = fmod(angle, two_pi);

  // Shift from [0, 2pi) to [-pi, pi).
  angle = remainder - M_PI;
}

