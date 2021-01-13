 #define _USE_MATH_DEFINES
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include "math.h"
#include "nanoflann.hpp"
#include <Eigen3/Eigen/Dense>
#include <chrono> 
#include <random>


using namespace std;
using namespace cv;
using namespace nanoflann;
using namespace std::chrono;


// ARGUMENTS
 //how many KNN neighbors to find (both for ICP and TR_ICP)
const int K = 5;

// max iterations after which both TR_ICP and ICP exit
const int MAX_ITERATIONS = 50;

// file into which the points of the final restult are saved
string OUTPUT_FILE = ("../data/OUTPUT.xyz");

// FOR ICP
// exit condition: after error change drops below this number, stop iterating
const double ERROR_DROP_THRESH = 0.0001;
const double ICP_ERROR_LOW_THRESH = 0.01;

// FOR TR_ICP
// exit condition: after distance change drops below this number, stop iterating
const double DIST_DROP_THRESH = 150.0;
// exit condition: after error drops below this number, stop iterating
const double ERROR_LOW_THRESH = 0.05;

// number of points the TR_ICP algortihm is trimmed to
int NUM_OF_TR_POINTS = 40000;

// if this param true, offsets the second pointcloud with static and gaussian offsets
// meant for testing if you just have one pointcloud on hand
// if set to 0, second pointcloud just imported as original
const bool APPLY_OFFSET_AND_GAUSSIAN_ON_SECOND_CLOUD = 1;
// ************************************

double prev_error = 0;

/**
 * @brief Performs the K nearest neighbors search between two 3d pointclouds
 *
 * @param cloud1      input pointcloud in which we search for neighbors
 * @param cloud2      input pointcloud through which we iterate and find k nearest neighbors from the cloud1
 * @param k           number of neighbors to be found
 * @param indices     output indices of the nearest neighbors
 * @param dists       output distances to the nearest neighbors
 * @param return_dist_squared   if this is true, returns square distances into dists. By default its false
 */
void searchNN(const Eigen::MatrixXf& pointcloud1,
    const Eigen::MatrixXf& pointcloud2,
    const size_t k,
    Eigen::MatrixXi& indices,
    Eigen::MatrixXf& dists,
    const bool return_dist_squared);

int import3dPointsFromFile(string file_path, vector<Point3d >& out_points);

/**
 * @brief Sorts the given matrix filled with float numbers. Outputs the sorted matrix and
 * the indexes they belong to in the old matrix
 *
 * @param m   input float matrix
 * @param out   output sorted float matrix
 * @param indexes   old indexes
 * @param to_save   how many elements to save in the new matrix and intro indexes, by default it saves all elements
 */
void sort_matr(const Eigen::MatrixXf& m,
    Eigen::MatrixXf& out,
    Eigen::MatrixXi& indexes,
    int to_save);

/**
 * @brief trimmed iterative closest point algorithm. Iteratevly moves points from src to dst
 *
 * @param src   input source points
 * @param dst   input destination points
 * @param src_trans   output resulting points at the end of iterations
 * @param max_itreations  exit condition: max number of iterations
 * @param error_low_thresh  exit condition: error btw points drops lower then this thresh
 * @param dist_drop_thresh  exit condition: distance change in iteration is lower then this thresh
 * @param Npo   number of points the ICP algortihm is trimmed to
 * @return int  returns 1 if tr_icp exited sucessfully, 0 if with error
 */
int tr_icp(const Eigen::MatrixXf& src,
    const Eigen::MatrixXf& dst,
    Eigen::MatrixXf& src_trans,
    const int max_itreations,
    const double error_low_thresh,
    const double dist_drop_thresh,
    const int Npo
);

/**
 * @brief iterative closest point algorithm. Iteratevly moves points from src to dst
 *
 * @param src   input source points
 * @param dst   input destination points
 * @param src_trans   output resulting points at the end of iterations
 * @param max_itreations  exit condition: max number of iterations
 * @param error_drop_thresh   exit condition: error change in iteration is lower then this thresh
 * @return int  returns 1 if icp exited sucessfully, 0 if with error
 */
int icp(const Eigen::MatrixXf& src,
    const Eigen::MatrixXf& dst,
    Eigen::MatrixXf& src_trans,
    const int max_itreations,
    const double error_drop_thresh);

/**
 * @brief estimates best transform between two matrices
 *
 * @param A   source matrix
 * @param B   destination matrix
 * @param R   return the rotation vector
 * @param t   return the translation vector
 * @return int  returns 1 if exited sucessfully, 0 if with error
 */
int best_fit_transform(const Eigen::MatrixXd& A,
    const Eigen::MatrixXd& B,
    Eigen::Matrix3d& R,
    Eigen::Vector3d& t);


/**
 * @brief estimates random rotation of matrix
 * 
 * @param matrix
 
Eigen::MatrixXf rotateRandomly(Eigen::MatrixXf& points2);
*/
// MAIN



int main(int argc, char** argv) {
   
    /*if (argc < 3) {
        cout << "Input format : Model point cloud, data point cloud" << endl;
        return -1;
    }*/

    // the containers in which we will store all the points
    vector<Point3d> points1;//Data-src
    vector<Point3d> points2;//Model-dst
     

    if (!import3dPointsFromFile(("../data/fountainDataSet.xyz"), points1)) {
        return 0;
    }

    Eigen::MatrixXf src(points1.size(), 3);
    for (int i = 0; i < points1.size(); i++) {
        src(i, 0) = points1[i].x;
        src(i, 1) = points1[i].y;
        src(i, 2) = points1[i].z;
    }

    ofstream outputFile3("../data/src.xyz");
    for (int g = 0; g < src.rows(); g++) {
        for (int gh = 0; gh < 3; gh++) {
            outputFile3 << src(g, gh) << " ";
        }
        outputFile3 << endl;
    }
    outputFile3.close();

     if (!import3dPointsFromFile(("../data/fountainModelSet.xyz"), points2)) {
        return 0;
    }
    Eigen::MatrixXf dst(points2.size(), 3);
    Eigen::MatrixXf noiseRotation = Eigen::MatrixXf::Zero(3, 3);

    //Add noisy rotation of 30 degs along z axis
    double angle = 30. * (M_PI / 180);
    noiseRotation(0, 0) = cos(angle);
    noiseRotation(0, 1) = -sin(angle);
    noiseRotation(0, 2) = 0;
    noiseRotation(1, 0) = sin(angle);
    noiseRotation(1, 1) = cos(angle);
    noiseRotation(1, 2) = 0;
    noiseRotation(2, 0) = 0;
    noiseRotation(2, 1) = 0;
    noiseRotation(2, 2) = 1.0;


    // if this param true, offsets the second pointcloud with static and gaussian offsets
    if (APPLY_OFFSET_AND_GAUSSIAN_ON_SECOND_CLOUD) {
        random_device rd; 
        normal_distribution<float> d(0, 0.1);
        mt19937 gen(rd());
      
        float sample;
        Eigen::MatrixXf data2 = Eigen::MatrixXf::Zero(3, 1);
        for (int i = 0; i < points2.size(); i++) {
            sample = d(gen);
            data2(0, 0) = points2[i].x;
            data2(1, 0) = points2[i].y;
            data2(2, 0) = points2[i].z;

            data2 = noiseRotation * data2;

            dst(i, 0) = data2(0, 0)+ sample;
            dst(i, 1) = data2(1, 0) + 1 + sample;
            dst(i, 2) = data2(2, 0) + 1 + sample;
            
        }
    }
    else {
        for (int i = 0; i < points2.size(); i++) {
            dst(i, 0) = points2[i].x;
            dst(i, 1) = points2[i].y;
            dst(i, 2) = points2[i].z;
        }
    }

    ofstream outputFile("../data/dst.xyz");
    for (int g = 0; g < dst.rows(); g++) {
        for (int gh = 0; gh < 3; gh++) {
            outputFile << dst(g, gh) << " ";
        }
        outputFile << endl;
    }
    outputFile.close();


    Eigen::MatrixXf out(dst.rows(), 3);
 
    auto start = high_resolution_clock::now();
    icp(src, dst, out, MAX_ITERATIONS, ERROR_DROP_THRESH);
   // tr_icp(src, dst, out, MAX_ITERATIONS, ERROR_LOW_THRESH, DIST_DROP_THRESH, NUM_OF_TR_POINTS);
   /*icp(src, dst, out, MAX_ITERATIONS, ERROR_DROP_THRESH);//Executed in  130891ms ms MSE: 0.287665/0.010000
    //tr_icp(src, dst, out, MAX_ITERATIONS, ERROR_LOW_THRESH, DIST_DROP_THRESH, NUM_OF_TR_POINTS);//Executed in 121563ms ms MSE  5.223052/0.050000
   // execute icp
    if (!icp(src, dst, out, MAX_ITERATIONS, ERROR_DROP_THRESH)) {
        cout << "Error while execution of ICP" << endl;
        return -1;
    }

    // // execute trimmed icp
    // if(!tr_icp(src, dst, out, MAX_ITERATIONS, ERROR_LOW_THRESH, DIST_DROP_THRESH, NUM_OF_TR_POINTS)){
    //   cout << "Error while execution of ICP" << endl;
    //   return -1;
    // }
    */
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    cout << "Finished in " + to_string(duration.count()) + "ms and saved result into: " + OUTPUT_FILE << endl;

    // save the result into output file
    ofstream outputFile1(OUTPUT_FILE);
    for (int g = 0; g < src.rows(); g++) {
        for (int gh = 0; gh < 3; gh++) {
            outputFile1 << out(g, gh) << " ";
        }
        outputFile1 << endl;
    }
    outputFile1.close();


    return 0;
}


void searchNN(const Eigen::MatrixXf& cloud1, const Eigen::MatrixXf& cloud2, const size_t k, Eigen::MatrixXi& indices, Eigen::MatrixXf& dists, const bool return_dist_squared = 0) {
    // Eigen::MatrixXf uses colMajor as default
    // copy the coords to a RowMajor matrix and search in this matrix
    // the nearest neighbors for each datapoint
    // Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> coords = cloud.leftCols(3);

    // different max_leaf values only affect the search speed 
    // and any value between 10 - 50 is reasonable
    const int max_leaf = 10;
    typedef Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> RowMatX3f;
    RowMatX3f coords1 = cloud1.leftCols(3);
    RowMatX3f coords2 = cloud2.leftCols(3);
    nanoflann::KDTreeEigenMatrixAdaptor<RowMatX3f> mat_index(3, coords1, max_leaf);
    mat_index.index->buildIndex();
    indices.resize(cloud2.rows(), k);
    dists.resize(cloud2.rows(), k);
    // do a knn search
    for (int i = 0; i < coords2.rows(); ++i) {
        std::vector<float> query_pt{ coords2.data()[i * 3 + 0], coords2.data()[i * 3 + 1], coords2.data()[i * 3 + 2] };

        std::vector<size_t> ret_indices(k);
        std::vector<float> out_dists_sqr(k);
        nanoflann::KNNResultSet<float> resultSet(k);
        resultSet.init(&ret_indices[0], &out_dists_sqr[0]);
        mat_index.index->findNeighbors(resultSet, &query_pt[0], nanoflann::SearchParams(10));
        for (size_t j = 0; j < k; ++j) {
            indices(i, j) = ret_indices[j];
            if (return_dist_squared) {//tr_icp
                dists(i, j) = out_dists_sqr[j];
            }
            else {
                dists(i, j) = std::sqrt(out_dists_sqr[j]);//icp
            }
        }
    }
}

int import3dPointsFromFile(string file_path, vector<Point3d >& out_points) {
    try {
        ifstream file(file_path);
        vector<Point3d > tmpv;
        Point3d tmp;

        double meh;

        while (file >> tmp.x && file >> tmp.y && file >> tmp.z && file >> meh && file >> meh && file >> meh) {
            // add a copy of tmp to points
            tmpv.push_back(tmp);
        }
        file.close();
        out_points = tmpv;
        return 1;
    }
    catch (const std::exception& e) {
        cout << "error opening the xyz file" << endl;
        return -1;
    }
}

int best_fit_transform(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, Eigen::Matrix3d& R, Eigen::Vector3d& t) {
  
    Eigen::Vector3d centroid_A(0, 0, 0);
    Eigen::Vector3d centroid_B(0, 0, 0);
    Eigen::MatrixXd AA = A;
    Eigen::MatrixXd BB = B;
    int row = A.rows();

    for (int i = 0; i < row; i++) {
        centroid_A += A.block<1, 3>(i, 0).transpose();
        centroid_B += B.block<1, 3>(i, 0).transpose();
    }
    centroid_A /= row;
    centroid_B /= row;
    for (int i = 0; i < row; i++) {
        AA.block<1, 3>(i, 0) = A.block<1, 3>(i, 0) - centroid_A.transpose();
        BB.block<1, 3>(i, 0) = B.block<1, 3>(i, 0) - centroid_B.transpose();
    }

    Eigen::MatrixXd H = AA.transpose() * BB;
    Eigen::MatrixXd U;
    Eigen::VectorXd S;
    Eigen::MatrixXd V;
    Eigen::MatrixXd Vt;

  /*
  JacobiSVD return U,S,V, S as a vector, "use U*S*Vt" to get original Matrix;
  */
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    U = svd.matrixU();
    S = svd.singularValues();
    V = svd.matrixV();
    Vt = V.transpose();

    R = Vt.transpose() * U.transpose();

    if (R.determinant() < 0) {
        Vt.block<1, 3>(2, 0) *= -1;
        R = Vt.transpose() * U.transpose();
    }

    t = centroid_B - R * centroid_A;

    return 1;

}

int icp(const Eigen::MatrixXf& src, const Eigen::MatrixXf& dst, Eigen::MatrixXf& src_trans, const int max_iterations, const double error_drop_thresh) {
    Eigen::MatrixXi indices;
    Eigen::MatrixXf dists;
    src_trans = src;
    float prev_dist_sum = FLT_MAX;

    // create a new matrix that is gonna be src, but ordered as closest to dst points - it has dst number of rows
    Eigen::MatrixXf src_neighbours(dst.rows(), 3);
    double mean_error = 0;

    float dist_sum = 0;
    // iterate through optimisation till you either reach max iterations or break out
    for (int i = 0; i < max_iterations; i++) {
        searchNN(src_trans, dst, K, indices, dists);

        // calculate error btw points src, dst
        dist_sum = 0;
        for (int d = 0; d < dists.size(); d++) {
            dist_sum += dists(d);
        }
        mean_error = dist_sum / dists.size();
        // check if error is dropping as fast as it should, if not, finish search
        if(abs(prev_error-mean_error) < error_drop_thresh){
             cout << "ICP has sucessfully reached the necesarry error_drop_thresh." << endl;
         return 1;
        }

        if (abs(mean_error) < ICP_ERROR_LOW_THRESH) {
            cout << "ICP has sucessfully reached the necesarry error_low_thresh." << endl;
            return 1;
        }

        // reorder src to fit to the nearest neighbour scheme
        for (int j = 0; j < src_neighbours.rows(); j++) {
            int ind = indices(j, 1);
            src_neighbours(j, 0) = src_trans(ind, 0);
            src_neighbours(j, 1) = src_trans(ind, 1);
            src_neighbours(j, 2) = src_trans(ind, 2);
        }

        // find transform matrix
        Eigen::Matrix3d tR;
        Eigen::Vector3d tt;
        best_fit_transform(src_neighbours.cast <double>(), dst.cast <double>(), tR, tt);
        Eigen::Matrix3f R = tR.cast<float>();
        Eigen::Vector3f t = tt.cast<float>();

        // rotation
        src_trans = (R * src_trans.transpose()).transpose();

        // translation
        for (int fs = 0; fs < src_trans.rows();fs++) {
            for (int a = 0; a < 3; a++) {
                src_trans(fs, a) = src_trans(fs, a) + t(a);
              
            }
        }

        cout << "********ICP Cycle " + to_string(i) + "*****" << endl;
        cout << "MSE: " + to_string(mean_error) + "/" + to_string(ICP_ERROR_LOW_THRESH) << endl;
        cout << "Change of MSE: " + to_string(abs(prev_error - mean_error)) + "/" + to_string(error_drop_thresh) << endl;

        // cout << to_string(abs(prev_dist_sum-dist_sum)) << endl;
        // cout << to_string(mean_error) << endl;

        prev_error = mean_error;
        prev_dist_sum = dist_sum;
    }
}

void sort_matr(const Eigen::MatrixXf& m,
    Eigen::MatrixXf& out,
    Eigen::MatrixXi& indexes,
    int to_save = -1) {
    vector<pair<float, int> > vp;

    // if to_save parameter is default (-1), all elements of the original matrix are saved
    // otherwise the number of elements that is passed is preserved 
    if (to_save == -1) {
        to_save = m.rows();
    }

    // Inserting element in pair vector 
    // to keep track of previous indexes 
    for (int i = 0; i < m.rows(); i++) {
        float vaaal = m(i);
        vp.push_back(make_pair(vaaal, i));
    }

    // Sorting pair vector 
    std::stable_sort(vp.begin(), vp.end(),
        [](const auto& a, const auto& b) {return a.first < b.first;});

    Eigen::MatrixXi ind(to_save, 1);
    Eigen::MatrixXf tmp_out(to_save, 1);
    for (int i = 0; i < to_save; i++) {
        tmp_out(i, 0) = m(vp[i].second);
        ind(i, 0) = vp[i].second;
    }

    out = tmp_out;
    indexes = ind;
}


int tr_icp(const Eigen::MatrixXf& src,
    const Eigen::MatrixXf& dst,
    Eigen::MatrixXf& src_trans,
    const int max_itreations,
    const double error_low_thresh,
    const double dist_drop_thresh,
    const int Npo
) {
    float prev_dist_sum = FLT_MAX;
    Eigen::MatrixXi knn_indices;
    Eigen::MatrixXf sq_dists, tr_sq_dists(Npo, 1);
    Eigen::MatrixXf tr_dst(Npo, 3), tr_src(Npo, 3);
    src_trans = src;

    Eigen::MatrixXf src_neighbours(dst.rows(), 3);
    double mean_error = 0;

    // iterate through optimisation till you either reach max iterations or break out
    for (int i = 0; i < max_itreations; i++) {
        searchNN(src_trans, dst, K, knn_indices, sq_dists, 1);


        // sort and trimm distances
        Eigen::MatrixXi old_dst_ind;
        Eigen::MatrixXf sorted_tr_distances;
        sort_matr(sq_dists, sorted_tr_distances, old_dst_ind, Npo);

        // save trimmed source and destination
        for (int i = 0; i < Npo; i++) {
            int dst_ind = old_dst_ind(i);
            int src_ind = knn_indices(dst_ind, 1);
            tr_src.block<1, 3>(i, 0) = src_trans.block<1, 3>(src_ind, 0);
            tr_dst.block<1, 3>(i, 0) = dst.block<1, 3>(dst_ind, 0);
        }

        // calculate stopping conditions ************************
        // sum up all the smallest distances
        float dist_sum = sq_dists.sum();  //sorted_tr_distances.sum();

        // trimmed mean square error
        float e = dist_sum / Npo;

        // if mse is lower then thresh or if distance drop if below the thresh, stop the tr_icp
        if (e < error_low_thresh || abs(prev_dist_sum - dist_sum) < dist_drop_thresh) {
            cout << "TR_ICP has sucessfully reached the boundary conditions." << endl;
            return 1;
        }

        // calculate translation ************************************
        // find transform matrix
        Eigen::Matrix3d tR;
        Eigen::Vector3d tt;
        best_fit_transform(tr_src.cast<double>(), tr_dst.cast<double>(), tR, tt);
        Eigen::Matrix3f R = tR.cast<float>();
        Eigen::Vector3f t = tt.cast<float>();

        // rotation
        src_trans = (R * src_trans.transpose()).transpose();

        // translation
        for (int fs = 0; fs < src_trans.rows();fs++) {
            for (int a = 0; a < 3; a++) {
                src_trans(fs, a) = src_trans(fs, a) + t(a);
            }
        }

        cout << "********TR_ICP Cycle " + to_string(i) + "*****" << endl;
        cout << "trimmed MSE: " + to_string(e) + "/" + to_string(error_low_thresh) << endl;
        cout << "Change of trimmed MSE: " + to_string(abs(prev_dist_sum - dist_sum)) + "/" + to_string(dist_drop_thresh) << endl;

        // cout << to_string(abs(prev_dist_sum-dist_sum)) << endl;
        // cout << to_string(e) << endl;

        prev_dist_sum = dist_sum;
    }

    return 1;
}

 
