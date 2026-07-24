#ifndef DATASET_
#define DATASET_

#include <array>
#include <string>
#include <fstream>
#include <iostream>
#include <algorithm>
using std::array;
using std::string;


class Dataset {
    private:
    // 
    int n, d;
    std::ifstream data_file;
    //Data as one memoryblock: data[i][j] written as data[i*d+j]
    float* data;
    bool selfset_data = false;
    
    public:
    explicit Dataset(int _n, int _d){
        n = _n;
        d = _d;
    
        data = new float[n*d];
        selfset_data = true;
    };
    explicit Dataset(int _n, int _d, float* _data){
        n = _n;
        d = _d;
    
        data = _data;
    };
    explicit Dataset(){
        n = 0;
        d = 0;
    
        data = new float[0];
    };
    
    ~Dataset() {    
        if (selfset_data) delete data; 
    };

    
    
    public:
    float get(int i, int j) {
        return data[i*d+j];
    };

    void set(int i, int j, float val) {
        data[i*d+j] = val;
    };

    float* get_data_pointer() {
        return data;
    }
    

    void load_datafile(string data_file_name){
        std::ifstream data_file(data_file_name.c_str());
        int data_count = std::count(std::istreambuf_iterator<char>(data_file),
                                std::istreambuf_iterator<char>(), '\n');
        if (data_count < n) {
            std::cout << "NOT ENOUGH DATA!! data_count: " << data_count << " n: " << n << "\n";
            exit(3);
        }

        data_file.clear();
        data_file.seekg(0, std::ios::beg);

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) {
                data_file >> data[i*d+j];
            }
        }
    };

    void print_datasample(int n, int d) {
        int max1 = std::min(10,n);
        int max2 = std::min(10,d);
        for (int i = 0; i < max1; i++) {
            for (int j = 0; j < max2; j++) {
                std::cout << data[i*d+j] << " ";
            }
            std::cout << "\n";
        }
        std::cout << std::endl;
    };
};
#endif