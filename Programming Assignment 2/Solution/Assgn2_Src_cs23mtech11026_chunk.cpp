#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <omp.h>

using namespace std;
using namespace std::chrono;

#define MAX 2048
int N, K, C, BT;

vector<vector<int>> matA(MAX, vector<int>(MAX, 0));
vector<vector<int>> matC(MAX, vector<int>(MAX, 0));

void *chunk(void *param)
{
    int thread_num = (long)param; // Extract thread number from parameter
    int chunksize = N / K;
    int end = (thread_num + 1) * chunksize;

    // Perform matrix multiplication for the assigned chunk
    for (int i = (thread_num - 1) * chunksize; i < end; i++)
    {
        for (int j = 0; j < N; j++)
        {
            for (int l = 0; l < N; l++)
            {
                matC[i][j] += matA[i][K] * matA[K][j];
            }
        }
    }
    pthread_exit(0);
}

int main()
{
    ifstream inputFile("inp.txt");
    if (!inputFile.is_open())
    {
        cerr << "Error opening the input file." << endl;
        return 1;
    }

    inputFile >> N >> K >> C >> BT;
    if (N <= 0 || K <= 0 || N > MAX || K > N)
    {
        cerr << "Invalid input parameters." << endl;
        return 1;
    }
    // Read matrix A from the input file
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            inputFile >> matA[i][j];
        }
    }
    inputFile.close();

    
    // Create K threads for chunk method
    pthread_t tid[K];         
    pthread_attr_t attr;      
    pthread_attr_init(&attr);

    // Calculate number of threads per core
    int threads_per_core = max(1, BT / C);

    // Set CPU affinity for bounded threads
    cpu_set_t cpuset;

    auto start_time_chunk = high_resolution_clock::now();
    for (int i = 0; i < K; i++)
    {
        long tmp = i + 1;
        pthread_create(&tid[i], &attr, chunk, (void *)(tmp));

        if (i < BT)
        {
            // Set CPU affinity for thread
            CPU_ZERO(&cpuset);
            CPU_SET((i / threads_per_core) % C, &cpuset);               
            pthread_setaffinity_np(tid[i], sizeof(cpu_set_t), &cpuset); 
        }
    }
    for (int i = 0; i < K; i++)
    {
        pthread_join(tid[i], NULL);
    }
    auto end_time_chunk = high_resolution_clock::now();
    auto time_taken_chunk = duration_cast<microseconds>(end_time_chunk - start_time_chunk);

    // Output time taken to out.txt
    ofstream outputFile("out_chunk.txt");
    if (!outputFile.is_open())
    {
        cerr << "Error opening the output file." << endl;
        return 1;
    }
    outputFile << "Square of Matrix A" << endl;
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            outputFile << matC[i][j] << " ";
        }
        outputFile << endl;
    }
    outputFile << "Time taken: " << time_taken_chunk.count() << " microseconds" << endl;
    outputFile.close();
    // Output time taken
    cout << "Time taken: " << time_taken_chunk.count() << " microseconds" << endl;

    return 0;
}

