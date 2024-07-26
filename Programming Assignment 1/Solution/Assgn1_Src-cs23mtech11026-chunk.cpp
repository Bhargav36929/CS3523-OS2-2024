#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <omp.h>

using namespace std;
using namespace std::chrono;

#define MAX 2048

vector<vector<int>> matA(MAX, vector<int>(MAX, 0));
vector<vector<int>> matC(MAX, vector<int>(MAX, 0));

int main()
{
    ifstream inputFile("inp2048.txt");
    if (!inputFile.is_open())
    {
        cerr << "Error opening the input file." << endl;
        return 1;
    }

    int N, K;
    inputFile >> N >> K;

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

    // Chunk size is the number of rows divided by the number of threads
    int chunkSize = N / K;

    auto start_time = high_resolution_clock::now();
    // Use OpenMP for parallelization
    #pragma omp parallel num_threads(K)
    {
        int tid = omp_get_thread_num(); // Get thread ID
        int startRow = tid * chunkSize;
        int endRow = (tid + 1) * chunkSize;

        // Perform matrix multiplication on the assigned chunk
        for (int i = startRow; i < endRow; i++)
        {
            for (int j = 0; j < N; j++)
            {
                for (int k = 0; k < N; k++)
                {
                    matC[i][j] += matA[i][k] * matA[k][j];
                }
            }
        }
    }
    auto end_time = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end_time - start_time);

    // Output time taken to out.txt
    ofstream outputFile("out2048.txt");
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
    outputFile << "Time taken: " << duration.count() << " microseconds" << endl;
    outputFile.close();

    // Output time taken
    cout << "Time taken: " << duration.count() << " microseconds" << endl;

    return 0;
}
