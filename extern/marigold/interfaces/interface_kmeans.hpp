#ifndef KMEANSSTRATEGY_H
#define KMEANSSTRATEGY_H
#include "../dataset.cpp"
#include <memory>

class KmeansStrategy {
    public:
      // pure virtual function
        virtual ~KmeansStrategy() = default;
        virtual int* run(Dataset* dataset) = 0;
        virtual void clear() = 0;
        virtual void init(int max_iter, int n, int d, int k, Dataset* dataset) = 0;
      
    private:

        //int max_iter;       // Maximum number of iterations
        //int d;      // Dimensions of each datapoint 
        //int n;    // Number of datapoints   
        //int k;   // Number of clusters
                          
};

#endif