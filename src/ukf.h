#ifndef UKF_H
#define UKF_H

#include "measurement_package.h"
#include "Eigen/Dense"
#include <vector>
#include <string>
#include <fstream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

class UKF {
public:

  // initially set to false, set to true in first call of ProcessMeasurement
  bool is_initialized_;

  // if this is false, laser measurements will be ignored (except for init)
  bool use_laser_;

  // if this is false, radar measurements will be ignored (except for init)
  bool use_radar_;

  // state vector: [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units and rad
  VectorXd x_;

  // state covariance matrix
  MatrixXd P_;

  // time when the state is true, in us
  long long time_us_;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  double std_a_;

  // Process noise standard deviation yaw acceleration in rad/s^2
  double std_yawdd_;

  // Laser measurement noise standard deviation position1 in m
  double std_laspx_;

  // Laser measurement noise standard deviation position2 in m
  double std_laspy_;

  // Radar measurement noise standard deviation radius in m
  double std_radr_;

  // Radar measurement noise standard deviation angle in rad
  double std_radphi_;

  // Radar measurement noise standard deviation radius change in m/s
  double std_radrd_ ;

  // Weights of sigma points
  VectorXd weights_;

  // State dimension
  int n_x_;

  // Augmented state dimension
  int n_aug_;

  // Sigma point spreading parameter
  double lambda_;

  MatrixXd R_radar_;

  MatrixXd R_lidar_;

  /**
   * Constructor
   */
  UKF();

  /**
   * Destructor
   */
  virtual ~UKF();

  /**
   * ProcessMeasurement
   * @param meas_package The latest measurement data of either radar or laser
   */
  void ProcessMeasurement(const MeasurementPackage& measurement);

  /**
   * Prediction Predicts sigma points, the state, and the state covariance
   * matrix
   * @param delta_t Time between k and k+1 in s
   */
  MatrixXd Prediction(double delta_t);

  MatrixXd PredictSigmaPoints(double delta_t);

  MatrixXd GenerateAugmentedSigmaPoints();

  void PredictMeanAndCovariance(const MatrixXd& Xsig_pred);

  /**
   * Updates the state and the state covariance matrix using a laser measurement
   * @param meas_package The measurement at k+1
   */
  void UpdateLidar(const MeasurementPackage& measurement, const MatrixXd& Xsig_pred);

  MatrixXd TransformSigmaPointsToLidarSpace(const MatrixXd& Xsig_pred);

  /**
   * Updates the state and the state covariance matrix using a radar measurement
   * @param meas_package The measurement at k+1
   */
  void UpdateRadar(const MeasurementPackage& measurement, const MatrixXd& Xsig_pred);

  MatrixXd TransformSigmaPointsToRadarSpace(const MatrixXd& Xsig_pred);

  VectorXd GetMeanPredictedMeasurement(const MatrixXd& Zsig, int n_z);

  MatrixXd CalculateMeasurementCovariance(const MatrixXd& Zsig,
                                          const VectorXd& z_pred,
                                          const MatrixXd& R,
                                          int n_z,
                                          const int* angle_index = nullptr);

  MatrixXd CalculateCrossCorrelation(const MatrixXd& Zsig,
                                     const VectorXd& z_pred,
                                     const MatrixXd& Xsig_pred,
                                     int n_z,
                                     const int* angle_index = nullptr);

  void UpdateFromMeasurement(const MatrixXd& Tc,
                             const VectorXd& z_pred,
                             const MatrixXd& S,
                             const VectorXd& raw_measurements,
                             const int* angle_index = nullptr);

  static void NormalizeAngle(double& angle);
};

#endif /* UKF_H */
