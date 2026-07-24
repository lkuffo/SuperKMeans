// Marigold k-means (Mortensen et al., VLDB 2023; AU-DIS/scalable-kmeans), vendored.
// Local changes vs upstream: double->float; seeded random init (seed=42); get_centroids()
// accessor; MG_SetLabel scratch deletes uncommented (leak fix); OpenMP on the assignment loop.
#include "../interfaces/interface_kmeans.hpp"
#include "../kmeans_utils/utils.cpp"
#include <algorithm>
#include <cstring>
#include <random>
#include <vector>

class MARIGOLDKmeansStrategy : public KmeansStrategy {
    public:
        int* run(Dataset* data) {
            int iter = 0;
            bool converged = false;
           
            Calculate_squared_botup(d, n, data_ptr, data_ss, l_pow);
            
            for (int i = 0; i < k; i++) {
                for (int j = i; j < k; j++) {
                    c_to_c[i][j] = 0; 
                    c_to_c[j][i] = 0;
                }
            }

            while ((iter < max_inter) && (!converged)) {
                //calculate square centroids
                Calculate_squared_botup(d, k, centroids, centroid_ss, l_pow);    

                //assign to centroids
                #pragma omp parallel for schedule(dynamic)
                for (int i = 0; i < n; i++) {
                    float val = near[labels[i]] < l_hamerly[i] ? l_hamerly[i] : near[labels[i]]; 
                    if (u_elkan[i] > val) {
                         MG_SetLabel(i); 
                    }
                }
                converged = Recalculate(data_ptr, centroids, old_centroids, cluster_count, labels, div, n, k, d, feature_cnt);
                if (!converged) {
                    //TODO: refactor location of .. you know the drill 
                    Update_bounds(data_ptr, centroids, c_to_c, centroid_ss, l_elkan, u_elkan, l_hamerly, labels, div, near, n, k, d, feature_cnt);                   
                    
                }
                iter++;
            }   

            for (int j = 0; j < k; j++) {
                std::cout << cluster_count[j] << " ";
            }
            std::cout << std::endl;
            std::cout << "Iter:" << iter << " Feature_cnt: " << feature_cnt << std::endl;
                

            return labels;
        };

        float* get_centroids() { return centroids; }

        void MG_SetLabel(const int x) {
            int l = 0;
            int *mask = new int[k];
            std::fill_n(mask, k, 1);

            float *dist = new float[k];
            for (int j = 0; j < k; j++) {
                dist[j] = data_ss[x][0]+centroid_ss[j][0];
            }
            
            
            float val;
            float UB, LB;
            
            int mask_sum = k;

            while (l <= L && mask_sum > 1) {
                for (int j = 0; j < k; j++) {
                    if (mask[j] != 1) continue; 

                    //Elkan prune
                    val = std::max(l_elkan[x][j], 0.5f * c_to_c[labels[x]][j]);
                    if (u_elkan[x] < val) {     //Elkan check
                        mask[j] = 0;            //Mark as pruned centroid
                    } else {
                        //DistToLevel params (int x, int c, int d, float data[], float centroids[], float* data_ss[], float* centroid_ss[], float* dots[], int l, int L)
                        DistToLevel_bot(x, j, d, data_ptr, centroids, data_ss, centroid_ss, l, L, dist[j], UB, LB, feature_cnt, l_pow);
                        LB = sqrt(std::max(0.0f, LB));
                        //if (LB > l_elkan[x][j]) {
                            
                            if (LB > l_elkan[x][j]) {
                                l_elkan[x][j] = LB; //Keep maximum LB per c
                            }   
                        //}
                        
                        UB = sqrt(std::max(0.0f, UB));
                        if (UB < u_elkan[x]) {
                            labels[x] = j;
                            u_elkan[x] = UB; //Keep minimum UB across c
                        }       
                    } 
                }
                mask_sum = 0;
                for (int j = 0; j < k; j++) {
                    mask_sum += mask[j];
                }
                l++;
            }

            delete[] mask;
            delete[] dist;
            //END: Updated labels, l_elkan[x][.], u_elkan[x]
        }

        //void clear() {}
        void clear() {
            for (int i = 0; i < n; i++) {
                delete[] l_elkan[i];
            }
            delete[] l_elkan;

            delete[] l_hamerly;
            
            delete[] u_elkan;

            delete[] near;

            delete[] div;

            
            for (int i = 0; i < k; i++) {
                delete[] c_to_c[i];
            }
            delete[] c_to_c; 

            

            
            for (int i = 0; i < n; i++) {
                delete[] data_ss[i];
            }
            delete[] data_ss;

            for (int i = 0; i < k; i++) {
                delete[] centroid_ss[i];
            }
            delete[] centroid_ss;

            delete[] labels;
         
            delete[] cluster_count;
               
            //Init centroids  
            delete[] centroids;
            delete[] old_centroids;
            
            delete[] l_pow;
        }

        void init(int _max_iter, int _n, int _d, int _k, Dataset* _data) {
            
            max_inter = _max_iter;
            n = _n;
            d = _d;
            k = _k;
            data_ptr = _data->get_data_pointer();
            feature_cnt = 0;

            //stepwise levels
            L = ceil(log10(d)/log10(4));

            //bounds
            l_elkan = new float*[n];
            for (int i = 0; i < n; i++) {
                l_elkan[i] = new float[k];
                std::fill(l_elkan[i], l_elkan[i]+k, 0.0);
            }

            l_hamerly = new float[n];
            std::fill(l_hamerly, l_hamerly+n, 0);


            u_elkan = new float[n];
            std::fill(u_elkan, u_elkan+n, std::numeric_limits<float>::max());

            
            near = new float[k];
            std::fill(near, near+k, 0);

            div = new float[k];

            //c_to_c
            c_to_c = new float*[k];//[new float[k]];
            for (int i = 0; i < k; i++) {
                c_to_c[i] = new float[k];
            }       

            l_pow = new int[L+1];
            for (int i = 0; i <= L; i++) {
                if (i == L && log10(d)/log10(4) < L) {
                    l_pow[i] = sqrt(d);
                } else {
                    l_pow[i] = int(pow(2,i));
                } 
            }

            
            //squared
            data_ss = new float*[n];
            for (int i = 0; i < n; i++) {
                data_ss[i] = new float[L+2];
            }

            centroid_ss = new float*[k];
            for (int i = 0; i < k; i++) {
                centroid_ss[i] = new float[L+2];
            }

            //Init labels
            labels = new int[n];
            std::fill(labels, labels+n, 0); 

            //Init cluster_counts
            cluster_count = new float[k];
            
            //Init centroids  
            centroids = new float[k*d];
            old_centroids = new float[k*d];
            
            //Initial centroids: k distinct data points chosen with a fixed seed (42)
            std::mt19937 gen(42);
            std::vector<int> perm(n);
            for (int i = 0; i < n; i++) perm[i] = i;
            for (int i = 0; i < k; i++) {
                std::uniform_int_distribution<int> pick(i, n - 1);
                std::swap(perm[i], perm[pick(gen)]);
                memcpy(centroids + (size_t) i * d, data_ptr + (size_t) perm[i] * d, sizeof(float) * d);
            }

        }
    private:

        int max_inter;
        int n;
        int d;
        int k;
        int L;
        float* centroids;
        float* old_centroids;
        float* cluster_count;
        long long feature_cnt;

        //float** dots;

        float** l_elkan;
        float* l_hamerly;
        float* u_elkan;


        int* l_pow;

        float* near;

        float* div;

        float** c_to_c;
        float** data_ss;
        float** centroid_ss;

        //x to c [x*k+c]
        int* labels;

        float* data_ptr;// = data->get_data_pointer();

};