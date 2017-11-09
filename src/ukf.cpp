#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // initially set to false, set to true in first call of ProcessMeasurement
  is_initialized_ = false;

  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(DIM_X);

  // initial covariance matrix
  P_ = MatrixXd(DIM_X, DIM_X);
  P_.setIdentity(DIM_X, DIM_X);

  ///* predicted sigma points matrix
  Xsig_aug_ = MatrixXd(DIM_AUG, DIM_SIGMA);
  Xsig_pred_ = MatrixXd(DIM_X, DIM_SIGMA);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 0.2;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.2;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.3;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */

  ///* State dimension
  n_x_ = DIM_X;

  ///* Augmented state dimension
  n_aug_ = DIM_AUG;

  ///* Sigma point spreading parameter
  lambda_ = 3.0 - DIM_X;

  ///* Weights of sigma points  
  weights_ = VectorXd(2*n_aug_+1);
  weights_.fill(0.5/(lambda_ + n_aug_));
  weights_(0) = lambda_ / (lambda_ + n_aug_);

  R_rad_ = MatrixXd(DIM_Z_RAD, DIM_Z_RAD);
  R_rad_ << std_radr_*std_radr_, 0.0, 0.0,
            0.0, std_radphi_*std_radphi_, 0.0,
            0.0, 0.0, std_radrd_*std_radrd_;

  R_las_ = MatrixXd(DIM_Z_LAS, DIM_Z_LAS);
  R_las_ << std_laspx_*std_laspx_, 0.0,
            0.0, std_laspy_*std_laspy_;

  Q_ = MatrixXd(DIM_NU, DIM_NU);
  Q_ << std_a_*std_a_, 0.0,
        0.0, std_yawdd_*std_yawdd_;

  n_nis_laser = 0;
  n_high_nis_laser = 0;
  total_nis_laser = 0.0;

  n_nis_radar = 0;
  n_high_nis_radar = 0;
  total_nis_radar = 0.0;
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */  
  cout << "ProcessMeasurement: "
      << " " << meas_package.sensor_type_ 
      << " " << meas_package.timestamp_ 
      << " " << meas_package.raw_measurements_.transpose() << endl;

  if ((meas_package.sensor_type_ == MeasurementPackage::RADAR && !use_radar_) ||
      (meas_package.sensor_type_ == MeasurementPackage::LASER && !use_laser_))
    return;

  if (!is_initialized_) {
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      return;
    } else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      x_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], 0, 0, 0;
    }

    time_us_ = meas_package.timestamp_;
    is_initialized_ = true;

    cout << "init x_ " << endl << x_ << endl;
    cout << "init P_ " << endl << P_ << endl;
    return;
  }

  Prediction((meas_package.timestamp_ - time_us_) / 1000000.0);

  if (meas_package.sensor_type_ == MeasurementPackage::RADAR)
    UpdateRadar(meas_package);
  else if (meas_package.sensor_type_ == MeasurementPackage::LASER)
    UpdateLidar(meas_package);

  cout << "x_ " << endl << x_ << endl;
  cout << "P_ " << endl << P_ << endl;

  time_us_ = meas_package.timestamp_;
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */
  int n_nu = n_aug_-n_x_;
  VectorXd x_aug = VectorXd(n_aug_);
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
 
  //create augmented state
  x_aug.head(n_x_) = x_;
  x_aug.tail(n_nu) = VectorXd::Zero(n_nu);
  //create augmented covariance matrix
  P_aug.fill(0);
  P_aug.block(0,0,n_x_,n_x_) = P_;
  P_aug.block(n_x_,n_x_,n_nu,n_nu) = Q_;  
  //create square root matrix
  MatrixXd A = P_aug.llt().matrixL();  
  //create augmented sigma points
  float d = sqrt(lambda_ + n_aug_);
  MatrixXd dA = d * A;
  MatrixXd xx = x_aug.replicate(1,n_aug_);  
  Xsig_aug_.col(0) = x_aug;
  Xsig_aug_.block(0,1,n_aug_,n_aug_) = xx + dA;
  Xsig_aug_.block(0,1+n_aug_,n_aug_,n_aug_) = xx - dA;

  //cout << "Xsig_aug_ " << endl << Xsig_aug_ << endl;

  //predict sigma points
  for(int i = 0; i < (2 * n_aug_ + 1); i++ ) {
      double p_x = Xsig_aug_(0,i);
      double p_y = Xsig_aug_(1,i);
      double v = Xsig_aug_(2,i);
      double psi = Xsig_aug_(3,i);
      double psi_d = Xsig_aug_(4,i);
      double nu_a = Xsig_aug_(5,i);
      double nu_psi_dd = Xsig_aug_(6,i);
      
      if(fabs(psi_d) > 0.001) {
        Xsig_pred_(0,i) = p_x + 
                v * (sin(psi + psi_d * delta_t) - sin(psi)) / psi_d + 
                0.5 * delta_t * delta_t * nu_a * cos(psi);
        Xsig_pred_(1,i) = p_y + 
                v * (-cos(psi + psi_d * delta_t) + cos(psi)) / psi_d + 
                0.5 * delta_t * delta_t * nu_a * sin(psi);
      } else {
        //avoid division by zero
        Xsig_pred_(0,i) = p_x + 
                v * cos(psi) * delta_t + 
                0.5 * delta_t * delta_t * nu_a * cos(psi);
        Xsig_pred_(1,i) = p_x + 
                v * sin(psi) * delta_t + 
                0.5 * delta_t * delta_t * nu_a * sin(psi);
      }
      
      Xsig_pred_(2,i) = v + delta_t * nu_a;
      Xsig_pred_(3,i) = psi + psi_d * delta_t + 0.5 * delta_t * delta_t * nu_psi_dd;
      Xsig_pred_(4,i) = psi_d + delta_t * nu_psi_dd;
  }

  //cout << "Xsig_pred_ " << endl << Xsig_pred_ << endl;
 
  //predict state mean
  x_ = Xsig_pred_ * weights_;  

  //predict state covariance matrix

  P_.fill(0);
  for(int i=0; i < (2 * n_aug_ + 1); i++){
      VectorXd error = Xsig_pred_.col(i) - x_;
      //cout << "i " << i << " error " << endl << error << endl;
      //angle normalization
      while (error(3)> M_PI) error(3)-=2.*M_PI;
      while (error(3)<-M_PI) error(3)+=2.*M_PI;
      P_ = P_ + weights_(i) * error * error.transpose();
  }

}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage& meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */

  int n_z = DIM_Z_LAS;

  //create matrix for sigma points in measurement space
  MatrixXd z_sig = MatrixXd(n_z, 2 * n_aug_ + 1);

  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  
  //measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z,n_z);

  //transform sigma points into measurement space
  for(int i=0; i < (2 * n_aug_ + 1); i++) {
      double p_x = Xsig_pred_(0,i);
      double p_y = Xsig_pred_(1,i);
      double v = Xsig_pred_(2,i);
      double psi = Xsig_pred_(3,i);
      double psi_d = Xsig_pred_(4,i);
      
      z_sig(0,i) = p_x;
      z_sig(1,i) = p_y;
  }
  //cout << "z_sig " << endl << z_sig << endl;

  //calculate mean predicted measurement
  z_pred = z_sig * weights_;
  //cout << "z_pred " << endl << z_pred << endl;
 
  //calculate measurement covariance matrix S
  S.fill(0);
  for(int i=0; i < (2 * n_aug_ + 1); i++) {
      VectorXd error = z_sig.col(i) - z_pred; 
      //cout << "i " << i << " error " << endl << error << endl;
      S = S + weights_(i) * error * error.transpose();
  }
       
  S = S + R_las_;

  //calculate cross correlation matrix
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  Tc.fill(0.0);
  for (int i = 0; i < (2 * n_aug_ + 1); i++) {  //2n+1 simga points

    //residual
    VectorXd z_diff = z_sig.col(i) - z_pred;

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  //Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  //residual
  VectorXd z = VectorXd(n_z);
  z << meas_package.raw_measurements_[0],
      meas_package.raw_measurements_[1];
  VectorXd z_diff = z - z_pred;

  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K*S*K.transpose();

  double nis = z_diff.transpose() * S.inverse() * z_diff;
  n_nis_laser++;
  n_high_nis_laser += ((nis > 5.991) ? 1 : 0);
  total_nis_laser += nis;

  cout << "laser nis " << nis 
      << " avg " << (total_nis_laser/n_nis_laser) 
      << " high % " << (100 * n_high_nis_laser / n_nis_laser) << endl;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage& meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */
  int n_z = DIM_Z_RAD;

  //create matrix for sigma points in measurement space
  MatrixXd z_sig = MatrixXd(n_z, 2 * n_aug_ + 1);

  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  
  //measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z,n_z);

  //transform sigma points into measurement space
  for(int i=0; i < (2 * n_aug_ + 1); i++) {
      double p_x = Xsig_pred_(0,i);
      double p_y = Xsig_pred_(1,i);
      double v = Xsig_pred_(2,i);
      double psi = Xsig_pred_(3,i);
      double psi_d = Xsig_pred_(4,i);
      
      z_sig(0,i) = sqrt(p_x*p_x + p_y*p_y);
      z_sig(1,i) = atan2(p_y, p_x);
      z_sig(2,i) = (p_x * v * cos(psi) + p_y * v * sin(psi)) / z_sig(0,i) ;
  }
  //cout << "z_sig " << endl << z_sig << endl;
  //calculate mean predicted measurement
  z_pred = z_sig * weights_;
  //cout << "z_pred " << endl << z_pred << endl;
  
  //calculate measurement covariance matrix S
  S.fill(0);
  for(int i=0; i < (2 * n_aug_ + 1); i++) {
      VectorXd error = z_sig.col(i) - z_pred; 
      //cout << "i " << i << " error " << endl << error << endl;
      //angle normalization
      while (error(1)> M_PI) error(1)-=2.*M_PI;
      while (error(1)<-M_PI) error(1)+=2.*M_PI;
      S = S + weights_(i) * error * error.transpose();
  }
       
  S = S + R_rad_;

  //calculate cross correlation matrix
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  Tc.fill(0.0);
  for (int i = 0; i < (2 * n_aug_ + 1); i++) {  //2n+1 simga points

    //residual
    VectorXd z_diff = z_sig.col(i) - z_pred;
    //angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  //Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  //residual
  VectorXd z = VectorXd(n_z);
  z << meas_package.raw_measurements_[0],
      meas_package.raw_measurements_[1],
      meas_package.raw_measurements_[2];
  VectorXd z_diff = z - z_pred;

  //angle normalization
  while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
  while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K*S*K.transpose();

  double nis = z_diff.transpose() * S.inverse() * z_diff;
  n_nis_radar++;
  n_high_nis_radar += ((nis > 5.991) ? 1 : 0);
  total_nis_radar += nis;

  cout << "radar nis " << nis 
      << " avg " << (total_nis_radar/n_nis_radar) 
      << " high % " << (100 * n_high_nis_radar / n_nis_radar) << endl;
}
