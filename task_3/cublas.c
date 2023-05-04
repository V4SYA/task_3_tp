#include "/opt/nvidia/hpc_sdk/Linux_x86_64/22.11/math_libs/11.8/targets/x86_64-linux/include/cublas_v2.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define BILLION 1000000000

double CORNER_1 = 10;
double CORNER_2 = 20;
double CORNER_3 = 30;
double CORNER_4 = 20;

int main(int argc, char** argv) {
    
    struct timespec start, stop;
    clock_gettime(CLOCK_REALTIME, &start);
    
    int n, iter_max;
    double tol;
    sscanf(argv[1], "%d", &n);
    sscanf(argv[2], "%d", &iter_max);
    sscanf(argv[3], "%lf", &tol);
    
    double *tmp;

    const double alpha = -1;
    double step1 = 10.0 / (n - 1);

    double* A = (double*)calloc(n*n, sizeof(double));
    double* Anew = (double*)calloc(n*n, sizeof(double));
    
    memset(A, 0, sizeof(double) * n * n);
    //Заполнение угловых значений сетки

    A[0] = CORNER_1;
    A[n-1] = CORNER_2;
    A[n * n - 1] = CORNER_3;
    A[n * (n - 1)] = CORNER_4;

//Создание копии массивов A и Anew на устройстве, а также копирование значений n и step1
#pragma acc enter data create(A[0:n*n], Anew[0:n*n]) copyin(n, step1)
//Указываем на начало параллельной области кода
#pragma acc kernels
    {
#pragma acc loop independent
        //Инициализация граничных значений массива A и Anew,
        //которые соответствуют граничным условиям задачи решения уравнения Пуассона
        for (int i = 1; i < n - 1; i++) {
			A[i] = CORNER_1 + i * step1;
			A[i * n] = CORNER_1 + i * step1;
			A[i * n + (n - 1)] = CORNER_2 + i * step1;
			A[n * (n - 1) + i] = CORNER_4 + i * step1;
		}
    }

    memcpy(Anew, A, sizeof(double) * n * n);

    cublasStatus_t status;
    cublasHandle_t handle;
    status = cublasCreate(&handle);

    int iter = 0;
    double error = 1.0;
    {
    while (iter < iter_max && error > tol) {
        iter++;
        if (iter % 100 == 0 || iter == 1) {
//Создаём копии массивов A и Anew на устройстве
#pragma acc data present(A[0:n*n], Anew[0:n*n])
//Начало параллельной области с указанием асинхронности выполнения
#pragma acc kernels async(1)
            {
//Распараллеливанием циклы по итерациям и потокам
#pragma acc loop independent collapse(2)
                for (int i = 1; i < n - 1; i++) {
                    for (int j = 1; j < n - 1; j++) {
                        Anew[i * n + j] = 0.25 * (A[(i + 1) * n + j] + 
                                                  A[(i - 1) * n + j] + 
                                                  A[i * n + j - 1] + 
                                                  A[i * n + j + 1]);
                    }
                }
            }
            int id = 0;
//Ожидание завершения асинхронных операций
#pragma acc wait
//Копируем данные с устройства на хост
#pragma acc host_data use_device(A, Anew)
            {
                cublasDaxpy(handle, n * n, &alpha, Anew, 1, A, 1);
                cublasIdamax(handle, n * n, A, 1, &id);

            }
//Копируем значения элемента с максимальным значением на хост
#pragma acc update self(A[id-1:1])
            error = fabs(A[id - 1]);
//Копируем данные с хоста на устройство
#pragma acc host_data use_device(A, Anew)
            cublasDcopy(handle, n * n, Anew, 1, A, 1);

        } else {
//Создаём копии массивов A и Anew на устройстве
#pragma acc data present(A[0:n*n], Anew[0:n*n])
//Указываем начало параллельной области с указанием асинхронности выполнения
#pragma acc kernels async(1)
            {
//Распараллеливаем циклы по итерациям и потокам
#pragma acc loop independent collapse(2)
                for (int i = 1; i < n - 1; i++) {
                    for (int j = 1; j < n - 1; j++) {
                        Anew[i * n + j] = 0.25 * (A[(i + 1) * n + j] + 
                                                  A[(i - 1) * n + j] + 
                                                  A[i * n + j - 1] + 
                                                  A[i * n + j + 1]);
                    }
                }
            }
        }

        tmp = A;
        A = Anew;
        Anew = tmp;
        
        //Отслеживаем прогресс вычислений
        if (iter % 100 == 0 || iter == 1)
#pragma acc wait(1)
            printf("%d %e\n", iter, error);

    }
}
    
    //Вывод результатов
    clock_gettime(CLOCK_REALTIME, &stop);
    double delta = (stop.tv_sec - start.tv_sec) + (double)(stop.tv_nsec - start.tv_nsec)/(double)BILLION;

    printf("%d\n", iter);
    printf("%0.6lf\n", error);
    printf("time %lf\n", delta);

    cublasDestroy(handle);
    return 0;
}
