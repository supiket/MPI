/*
Student Name: Bengisu Ozaydin
Student Number: 2018400285
Compile Status: Compiling
Program Status: Working
Notes : -
*/

#include <iostream>
#include <mpi.h>
#include <stdlib.h>
#include <fstream>
#include <vector>
#include <numeric>
#include <set>
#include <algorithm>

using namespace std;

// sort by ascending while keeping track of indices
void sort_ascending(double *W, int *indices, int A){
  double temp_W;
  double temp_indices;
  for(int i = 0; i < A; i++){
    for(int j = i+1; j < A; j++){
      if(W[i]> W[j]){
        temp_W = W[i];
        W[i] = W[j];
        W[j] = temp_W;
        temp_indices = indices[i];
        indices[i] = indices[j];
        indices[j] = temp_indices;
      }
    }
  }
}

// find manhattan distance between two arrays
double manhattan_distance(double features_1[], double features_2[], int size){
  double distance = 0;
  for(int i = 0; i < size-1; i++){
    distance += fabs(features_1[i] - features_2[i]);
  }
  return distance;
}

int main(int argc, char *argv[])
{
  int rank;
  int size;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // P: number of processors, one of which is master and other P-1 are slaves
  int P;
  ifstream input_file(argv[1]);
  input_file >> P;

  // N: number of instances
  // A: number of features
  // M: iteration count for weight updates
  // T: resulting number of features
  int N, A, M, T;
  input_file >> N >> A >> M >> T;
  int num_instances_per_slave = N/(P-1);
  double input_data[N][A+1];

  // master
  if (rank == 0) {
    // read feature values
    for(int i = 2; i < N + 2; i++){
      for(int j = 0; j < A + 1; j++){
        input_file >> input_data[i-2][j];
      }
    }
    input_file.close();

    // send data array to the slaves
    for(int slave_rank = 1; slave_rank <= size-1; slave_rank++){
      for(int instance_index = 0; instance_index < num_instances_per_slave; instance_index++){
        MPI_Send(input_data[(slave_rank-1)*num_instances_per_slave+instance_index], A+1, MPI_DOUBLE, slave_rank, 0, MPI_COMM_WORLD);
      }
    }

    // receive indices from slaves
    vector<int> all_indices;
    for(int slave_rank = 1; slave_rank <= size-1; slave_rank++){
      int indices[A];
      MPI_Recv(&indices[0], A, MPI_INT, slave_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      for(int i = A-T; i < A; i++){
        all_indices.push_back(indices[i]);
      }
    }

    // turn vector into set to get rid of duplicates. elements are already sorted
    set<int> all_indices_set(all_indices.begin(), all_indices.end());
    set<int>::iterator it = all_indices_set.begin();

    // console output of master
    cout << "Master P" << rank << " :";
    while(it != all_indices_set.end()){
      cout << " " << (*it);
      it++;
    }
    cout << endl;
   }

  // slave
  else{
    int index_helper = (rank-1)*num_instances_per_slave;
    // receive data array from master
    for(int instance_index = 0; instance_index < num_instances_per_slave; instance_index++){
      MPI_Recv(input_data[index_helper+instance_index], A+1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    // array of weights of features
    double W[A];

    // relief algorithm
    // M iterations
    for(int iterator = 0; iterator < M; iterator++){
      int selected = iterator % num_instances_per_slave;
      int class_value = input_data[index_helper+selected][A];
      double min_distance_hit = INT_MAX;
      double min_distance_miss = INT_MAX;
      int hit_index = -1;
      int miss_index = -1;

      // find closest hit and miss indices
      for(int i = 0; i < num_instances_per_slave; i++){
        if(i != selected){
          double distance = manhattan_distance(input_data[index_helper+selected], input_data[index_helper+i], A+1);
          if(input_data[index_helper+i][A] == class_value){ // possible hit
            if(distance < min_distance_hit){
              min_distance_hit = distance;
              hit_index = i;
            }
          }
          else{ // possible miss
            if(distance < min_distance_miss){
              min_distance_miss = distance;
              miss_index = i;
            }
          }
        }
      }

      // update W vector
      for(int feature_index = 0; feature_index < A; feature_index++){
        double min_feature = INT_MAX;
        double max_feature = INT_MIN;
        for(int i = 0; i < num_instances_per_slave; i++){
          double value = input_data[index_helper+i][feature_index];
          if(value < min_feature){
            min_feature = value;
          }
          if(value > max_feature){
            max_feature = value;
          }
        }
        double value_hit = fabs(input_data[index_helper+selected][feature_index] - input_data[index_helper+hit_index][feature_index]);
        double value_miss = fabs(input_data[index_helper+selected][feature_index] - input_data[index_helper+miss_index][feature_index]);
        double diff_hit = value_hit/(max_feature-min_feature);
        double diff_miss = value_miss/(max_feature-min_feature);
        W[feature_index] = W[feature_index]-diff_hit/M + diff_miss/M;
      }
    }

    // find T elements with highest weights
    int indices[A];
    for(int i = 0; i < A; i++){
      indices[i] = i;
    }
    sort_ascending(W, indices, A);
    sort(indices+A-T, indices+A);
    // slaves console output
    cout << "Slave P" << rank << " :";
    for(int i = A-T; i < A; i++){
      cout << " " << indices[i];
    }
    cout << endl;
    // send sorted indices to master
    MPI_Send(&indices[0], A, MPI_INT, 0, 0, MPI_COMM_WORLD);
  }

  MPI_Finalize();
  return(0);
}
