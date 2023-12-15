#include <iostream>
#include <chrono>
#include <immintrin.h>
using namespace std;

const int ARRAY_SIZE = 350; // размер массива
float arr1[ARRAY_SIZE * ARRAY_SIZE];
float arr2[ARRAY_SIZE * ARRAY_SIZE];
float arrRes[ARRAY_SIZE * ARRAY_SIZE];

//GEMM определяется как операция C = αAB + βC, где A и B являются входными матрицами, α и β — скалярными входными данными, а C — уже существующей матрицей, которая перезаписывается выходными данными.Простое матричное произведение AB представляет собой GEMM с α, равным единице, и β, равным нулю.

void gemm_v0(int M, int N, int K, const float* A, const float* B, float* C)
{
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            //Инициализация i * N + j-го элемента нулем
            C[i * N + j] = 0;
            //cout << "C[i * N + j] | i * N + j = " << C[i * N + j] << " | " << i * N + j << endl;
            //Получение i * N + j-го элемента матрицы
            for (int k = 0; k < K; ++k)
                //элемент += прохождение_по_элементам_строки * прохождение_по_элементам_столбца
                C[i * N + j] += A[i * K + k] * B[k * N + j];
        }
    }
}

void gemm_v1(int M, int N, int K, const float* A, const float* B, float* C)
{
    for (int i = 0; i < M; ++i)
    {
        // Присвоение адреса первого элемента i-й строки
        float* c = C + i * N;
        for (int j = 0; j < N; ++j)
            c[j] = 0;
        for (int k = 0; k < K; ++k)
        {
            // Ссылка на k-ю строку
            const float* b = B + k * N;
            // k-ый элемент i-й строки
            float a = A[i * K + k];
            // Проходимся по элементам строки и прибавляем
            // к нему (ЭС) j-й элемент из матрицы b (т.е. k-й строки)
            for (int j = 0; j < N; ++j) // Прибавляем 
                c[j] += a * b[j];
        }
    }
}

void gemm_v2(int M, int N, int K, const float* A, const float* B, float* C)
{
    for (int i = 0; i < M; ++i)
    {
        float* c = C + i * N;
        for (int j = 0; j < N; j += 8)
            _mm256_storeu_ps(c + j + 0, _mm256_setzero_ps());
        for (int k = 0; k < K; ++k)
        {
            const float* b = B + k * N;
            __m256 a = _mm256_set1_ps(A[i * K + k]);
            for (int j = 0; j < N; j += 16)
            {
                _mm256_storeu_ps(c + j + 0, _mm256_fmadd_ps(a,
                    _mm256_loadu_ps(b + j + 0), _mm256_loadu_ps(c + j + 0)));
                _mm256_storeu_ps(c + j + 8, _mm256_fmadd_ps(a,
                    _mm256_loadu_ps(b + j + 8), _mm256_loadu_ps(c + j + 8)));
            }
        }
    }
}


void consoleLogArr(float* arr, int n) {
    for (int i = 0; i < n; i++)
    {
        std::cout << arr[i] << std::endl;
    }
}

void generateRandomArray(float* arr) {
    for (int counter = 0; counter < ARRAY_SIZE; counter++)
    {
        arr[counter] = rand() % 50 - rand() % 50; // заполняем массив случайными значениями в диапазоне от -49 до 49 включительно
        // cout << array1[counter] << " "; // печать элементов одномерного массива array1
    }
}

int main()
{
    setlocale(LC_ALL, "Russian");
    chrono::steady_clock::time_point startTime;
    chrono::steady_clock::time_point endTime;
    chrono::duration<double> time_span;

    //Проверка умножения
    //float arr1[]{ 5, 2, 3, 1 };
    //float arr2[]{ 4, 6, 5, 2 };
    //Ответ: 30 17 34 20
    //float arrRes[]{ 0, 0, 0, 0 };
    //gemm_v1(2, 2, 2, arr1, arr2, arrRes);
    //consoleLogArr(arrRes, 4);

    generateRandomArray(arr1);
    generateRandomArray(arr2);

    //unsigned int startTime = clock(); // количество тактов процессора
    startTime = chrono::steady_clock::now();
    gemm_v0(ARRAY_SIZE, ARRAY_SIZE, ARRAY_SIZE, arr1, arr2, arrRes);
    //unsigned int endTime = clock(); // количество тактов процессора
    endTime = chrono::steady_clock::now();
    //double seconds = (double)(startTime - endTime); // количество тактов процессора
    time_span = chrono::duration_cast<chrono::duration<double>>(endTime - startTime);

    cout << "Время выполнения gemm_v0: " << time_span.count() << endl;


    generateRandomArray(arr1);
    generateRandomArray(arr2);
    startTime = chrono::steady_clock::now();
    gemm_v2(ARRAY_SIZE, ARRAY_SIZE, ARRAY_SIZE, arr1, arr2, arrRes);
    //unsigned int endTime = clock(); // количество тактов процессора
    endTime = chrono::steady_clock::now();
    time_span = chrono::duration_cast<chrono::duration<double>>(endTime - startTime);

    cout << "Время выполнения gemm_v2: " << time_span.count() << endl;
}


