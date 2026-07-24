#ifndef UTILS_CPP
#define UTILS_CPP
#include <cfloat>
#include <cmath>
#include <tuple>
#include <algorithm>
#include <iostream>
//#include <memory>
#include <cstring>

//TODO: In a pefect world this file does not exist and theese functions nicely exist within relevant strategies.

float Euclidian_distance(int x, int c, int d, int k, float data[], float centroids[], long long &feature_cnt) {
    //x datapoint index
    //c centroid index

    //d features/dimension
    float dist = 0;
    for (int j = 0; j < d; j++) {
        dist += (data[x*d+j]-centroids[c*d+j])*(data[x*d+j]-centroids[c*d+j]);
    };
    //std::cout << dist << std::endl;
    feature_cnt += d;
    if(dist < 0.0) dist = 0.0;
    return sqrt(dist);
}

float Squared_euclidian_distance(int x, int c, int d, int k, float data[], float centroids[], long long &feature_cnt) {
    //x datapoint index
    //c centroid index

    //d features/dimension
    float dist = 0;
    for (int j = 0; j < d; j++) {
        dist += (data[x*d+j]-centroids[c*d+j])*(data[x*d+j]-centroids[c*d+j]);
    };
    //std::cout << dist << std::endl;
    feature_cnt += d;
    if(dist < 0.0) dist = 0.0;
    return dist;
}


                                                                        //TODO: This parsing is insane and can be cleaned clean up, by defining this and dist to level within a strategy scope.   
void Update_bounds(float data[], float centroids[], float* c_to_c[], float* centroids_ss[], float* l_elkan[], float u_elkan[], float l_hamerly[], int labels[], float div[], float near[], int n, int k, int d, long long &feature_cnt) {
    //For all x in X
    for (int i = 0; i < n; i++) { 
        int smallest_id = labels[i] == 0 ? 1 : 0; 
        //For all c in C
        for (int j = 0; j < k; j++) {
            //l_elkan(x, c) <-- max{0, l_elkan(x, c) - div[c]} 
            float val = l_elkan[i][j]-div[j];  
            l_elkan[i][j] = 0 < val ? val : 0;
            if ((labels[i] != j) && (l_elkan[i][j] <= l_elkan[i][smallest_id])) {
                smallest_id = j; 
            }  
        }
        //u_elkan <- u_elkan + div[a[x]]
        u_elkan[i] += div[labels[i]];
        //l_hamerly <- min{l_elkan(x, c != a[x])}
        l_hamerly[i] = l_elkan[i][smallest_id];
    }
    //Calculate centroid to centroid distances
    for (int i = 0; i < k; i++) {
        c_to_c[i][i] = 0;
        
        for (int j = i+1; j < k; j++) {
            float tmp = 0; //centroids_ss[i][0] + centroids_ss[j][0];
            for (int f = 0; f < d; f++) {
                //TODO: this does not use squares when it could, ot could it?
                tmp += ((centroids[i*d+f] - centroids[j*d+f]) *
                    (centroids[i*d+f] - centroids[j*d+f]));
            }
            //feature_cnt += d;
            if(tmp < 0.0) tmp = 0.0;
            tmp = sqrt(tmp);

            
            //We can save distances for later use
            c_to_c[i][j] = (tmp);
            // THEY'RE THE SAME
            c_to_c[j][i] = c_to_c[i][j];
        }
        float smallest = DBL_MAX;
        float near_id = -1 ;
        for (int j = 0; j < k; j++) {
           if (i == j) continue;
           if (c_to_c[i][j] < smallest) {  
                smallest = c_to_c[i][j];  
                near[i] = 0.5 * smallest;
                near_id = j;
                //near[j] = 0.5 * smallest;
            } 
        }
        //std::cout << i << " is near " <<  near_id << "\n";
    
    }

    //END: Updated l_elkan, u_elkan, l_hamerly, near, c_to_c
};


bool Recalculate(const float data[], float centroids[], float old_centroids[], float cluster_count[], int labels[], float div[], const int n, const int k, const int d, long long &feature_cnt) {
    bool converged = true;
    //With the power of clever indexing we can use memcpy, to save old centroids. (If we are even smarter we swap the pointers and save a pass)   
    memcpy(old_centroids, centroids, sizeof(float)*k*d);
    //Wipe memory for new centroid calculations
    memset(centroids, 0.0, sizeof(float)*k*d);

    memset(cluster_count, 0, sizeof(float)*k);
    
    //Count size of clusters and add pos to centroid
    for (int i = 0; i < n; i++) {
        cluster_count[labels[i]]++;
        for (int j = 0; j < d; j++) { 
            centroids[labels[i]*d+j] += data[i*d+j];
        }
    }

    //Calculate new centroid positions
    for (int i = 0; i < k; i++) {  
        if (cluster_count[i] > 0) {
            for (int j = 0; j < d; j++) {
                centroids[i*d+j] /= cluster_count[i]; 
            }
        } else {
            for (int j = 0; j < d; j++) {
                centroids[i*d+j] = old_centroids[i*d+j];
            }
        }
    }

    //for (int i = 0; i < k; i++) {      
    //    std::cout << cluster_count[i] << " "; 
    //    }
    //std::cout << std::endl;

    //calculate div
    for (int j = 0; j < k; j++) {
        float tmp = 0.0;
        for (int f = 0; f < d; f++) {
            tmp += ((centroids[j*d+f] - old_centroids[j*d+f]) *
                    (centroids[j*d+f] - old_centroids[j*d+f]));
        }
        feature_cnt += d;
        if(tmp < 0.0) tmp = 0.0;
        div[j] = sqrt(tmp);
        if (div[j] > 0) { //This convergence check is slow and can possibly be done better using the old check
            converged = false;
        }
    }
    //END: div[.] updated
    return converged;
}


std::tuple<float, float> DistToLevel(const int x, const int c, const int d, const float data[], const float centroids[], const float *const data_ss[], const float *const centroid_ss[], const int l, const int L, float* dots[], float &UB, float &LB, long long &feature_cnt) {
    //Calculate dots  
    int d_sqrt = sqrt(d);
    //L is an int of log4(d), hence rounded down. 
    //Hence when log4(d) is not in natural, d_sqrt will be bigger and the correct end for the final level instead of 2^l.

    /*dots[x][c] = 0;
    for (int l_ = 0; l_ < std::min(d_sqrt,(int) pow(2,l)); l_++) {
        for (int l_2 = 0; l_2 < std::min(d_sqrt,(int) pow(2,l)); l_2++) {
            dots[x][c] += data[x*d+l_*d_sqrt+l_2]*centroids[c*d+l_*d_sqrt+l_2];
        }
    }*/
    int pow_ll = pow(2,l-1);
    int pow_l = pow(2,l);
    
    if (l==0) {
        dots[x][c] = data[x*d+0]*centroids[c*d+0];
    } else {
        //dots saved from previous level, hence only add dots from this level.
        //adding new cols from from known rows
        for (int l_ = 0; l_ < pow_ll; l_++) {
            for (int l_2 = pow_ll; l_2 < pow_l ; l_2++) {
                dots[x][c] += data[x*d+l_*d_sqrt+l_2]*centroids[c*d+l_*d_sqrt+l_2]; 
            }
        }
        //TODO: sqrt stuff for d != 2^x
        //add full new rows
        for (int l_ = pow_ll; l_ < pow_l; l_++) {
            for (int l_2 = 0; l_2 < pow_l; l_2++) {
                dots[x][c] += data[x*d+l_*d_sqrt+l_2]*centroids[c*d+l_*d_sqrt+l_2]; 
            }
        }   
    }
    feature_cnt += (2*pow_l)-(2*pow_ll);
    
    
    float dist = data_ss[x][L] + centroid_ss[c][L] - 2*dots[x][c]; 

    float margin = 2 * sqrt((data_ss[x][L]-data_ss[x][l])*(centroid_ss[c][L]-centroid_ss[c][l]));


    
    LB = dist - margin;//sqrt(std::max(0.0,dist - margin));
    UB = dist + margin;//sqrt(std::max(0.0,dist + margin));

    return {LB, UB};
};

std::tuple<float, float> DistToLevel_bot(const int x, const int c, const int d, const float data[], const float centroids[], const float *const data_ss[], const float *const centroid_ss[], const int l, const int L, float &dist, float &UB, float &LB, long long &feature_cnt, const int l_pow[]) {
    //Calculate dots  
    int d_sqrt = sqrt(d);
    //L is an int of log4(d), hence rounded down. 
    //Hence when log4(d) is not in natural, d_sqrt will be bigger and the correct end for the final level instead of 2^l.

    /*dots[x][c] = 0;
    for (int l_ = 0; l_ < std::min(d_sqrt,(int) pow(2,l)); l_++) {
        for (int l_2 = 0; l_2 < std::min(d_sqrt,(int) pow(2,l)); l_2++) {
            dots[x][c] += data[x*d+l_*d_sqrt+l_2]*centroids[c*d+l_*d_sqrt+l_2];
        }
    }*/
    //int pow_ll = pow(2,l-1);
    //int pow_l = pow(2,l);
    //l_pow[l]
    
    
    if (l==0) {
        dist -= 2*data[x*d+0]*centroids[c*d+0];
    } else {
        //dots saved from previous level, hence only add dots from this level.
        //adding new cols from from known rows
        for (int l_ = 0; l_ < l_pow[l-1]; l_++) {
            for (int l_2 = l_pow[l-1]; l_2 < l_pow[l] ; l_2++) {
                dist -= 2*data[x*d+l_*d_sqrt+l_2]*centroids[c*d+l_*d_sqrt+l_2]; 
            }
        }
        //TODO: sqrt stuff for d != 2^x
        //add full new rows
        for (int l_ = l_pow[l-1]; l_ < l_pow[l]; l_++) {
            for (int l_2 = 0; l_2 < l_pow[l]; l_2++) {
                dist -= 2*data[x*d+l_*d_sqrt+l_2]*centroids[c*d+l_*d_sqrt+l_2]; 
            }
        }   
    }
    #pragma omp atomic
    feature_cnt += (2*l_pow[l])-(2*l_pow[l-1]);

    float margin = 2 * sqrt((data_ss[x][l+1])*(centroid_ss[c][l+1]));


    LB = dist - margin;//sqrt(std::max(0.0,dist - margin));
    UB = dist + margin;//sqrt(std::max(0.0,dist + margin));

    return {LB, UB};
};






void Calculate_squared(const int d, const int elements, const float raw[], float* squared[], const int l_pow[]) {
    int L = log10(d)/log10(4);
    int d_sqrt = sqrt(d);
    int two_p_level_m1, two_p_level;
   
    for (int e = 0; e < elements; e++) {
        squared[e][0] = raw[e*d+0] * raw[e*d+0];

        for (int l = 1; l <= L; l++){
            two_p_level_m1 = l_pow[l-1];// pow(2,l-1);
            two_p_level = l_pow[l];
            squared[e][l] = squared[e][l-1];
            //dots saved from previous level, hence only add dots from this level.
            //adding new cols from from known rows
            for (int l_ = 0; l_ < two_p_level_m1; l_++) {
                for (int l_2 = two_p_level_m1; l_2 < two_p_level; l_2++) {
                    squared[e][l] += raw[e*d+l_*d_sqrt+l_2]*raw[e*d+l_*d_sqrt+l_2]; 
                }
            }
            //TODO: sqrt stuff for d != 2^x
            //add full new rows
            for (int l_ = two_p_level_m1; l_ < two_p_level; l_++) {
                for (int l_2 = 0; l_2 < two_p_level; l_2++) {
                    squared[e][l] += raw[e*d+l_*d_sqrt+l_2]*raw[e*d+l_*d_sqrt+l_2]; 
                }
            }
      
        } 
    }
}


void Calculate_squared_botup(int d, int elements, float raw[], float* squared[], const int l_pow[]) {
    int L = ceil(log10(d)/log10(4));
    int d_sqrt = sqrt(d);
    int two_p_level_m1, two_p_level;
    int bot_start = L;
    
    for (int e = 0; e < elements; e++) {
        squared[e][L+1] = 0; 

        for (int l = bot_start; l > 0; l--){
            two_p_level_m1 = l_pow[l-1];// pow(2,l-1);
            two_p_level = l_pow[l];

            squared[e][l] = squared[e][l+1];
            //dots saved from previous level, hence only add dots from this level.
            //adding new cols from from known rows
            for (int l_ = 0; l_ < two_p_level_m1; l_++) {
                for (int l_2 = two_p_level_m1; l_2 < two_p_level; l_2++) {
                    squared[e][l] += raw[e*d+l_*d_sqrt+l_2]*raw[e*d+l_*d_sqrt+l_2]; 
                }
            }
            //TODO: sqrt stuff for d != 2^x
            //add full new rows
            for (int l_ = two_p_level_m1; l_ < two_p_level; l_++) {
                for (int l_2 = 0; l_2 < two_p_level; l_2++) {
                    squared[e][l] += raw[e*d+l_*d_sqrt+l_2]*raw[e*d+l_*d_sqrt+l_2]; 
                }
            }
      
        }
        squared[e][0] = squared[e][1] + raw[e*d+0]*raw[e*d+0] ;
    }
}

#endif