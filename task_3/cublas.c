/*Данный код решает уравнение Пуассона на прямоугольной области с помощью метода Якоби и ускоряет вычисления с помощью директив OpenACC.
Вначале определяются параметры модели, такие как точность, размер сетки и максимальное число итераций.
Затем выделяется память на хосте и девайсе для двух массивов A и Anew,
которые хранят значения температуры на сетке.

Основной алгоритм выполняется до тех пор, пока значение переменной error не станет меньше заданного порога.
На каждой итерации цикла происходит вычисление новых значений на текущей точке с использованием значений
соседних точек и обновление значения error. Когда значение error становится меньше заданного порога,
цикл прерывается и программа выводит результаты решения уравнения Пуассона на экран*/

#include "/opt/nvidia/hpc_sdk/Linux_x86_64/21.11/math_libs/11.5/targets/x86_64-linux/include/cublas_v2.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#define BILLION 1000000000

int main(int argc, char** argv) {
    
    struct timespec start, stop;
    clock_gettime(CLOCK_REALTIME, &start);
    
    int n, iter_max;
    double tol;
    sscanf(argv[1], "%d", &n);
    sscanf(argv[2], "%d", &iter_max);
    sscanf(argv[3], "%lf", &tol);
    
    double *tmp;
    cublasHandle_t handle;
    cublasCreate(&handle);

    const double alpha = -1;
    double step1 = 10.0 / (n - 1);

    double* A = (double*)calloc(n*n, sizeof(double));
    double* A_new = (double*)calloc(n*n, sizeof(double));
    
    double x1 = 10.0, x2 = 20.0, y1 = 20.0, y2 = 30.0;
    
    //Заполнение угловых значений сетки
    A[0] = A_new[0] = x1;
    A[n] = A_new[n] = x2;
    A[n * (n - 1) + 1] = A_new[n * (n - 1) + 1] = y1;
    A[n * n] = A_new[n * n] = y2;

//Создание копии массивов A и A_new на устройстве, а также копирование значений n и step1
#pragma acc enter data create(A[0:n*n], A_new[0:n*n]) copyin(n, step1)
//Указываем на начало параллельной области кода
#pragma acc kernels
    {
#pragma acc loop independent
        //Инициализация граничных значений массива A и A_new,
        //которые соответствуют граничным условиям задачи решения уравнения Пуассона
        for (int i = 0; i < n; i++) {
            A[i*n] = A_new[i*n] = x1 + i * step1;
            A[i] = A_new[i] = x1 + i * step1;
            A[(n - 1) * n + i] = A_new[(n - 1) * n + i] = y1 + i * step1;
            A[i * n + (n - 1)] = A_new[i * n + (n - 1)] = x2 + i * step1;        }
    }

    int iter = 0;
    double error = 1.0;
    {
    while (iter < iter_max && error > tol) {
        iter++;
        if (iter % 100 == 0 || iter == 1) {
//Создаём копии массивов A и A_new на устройстве
#pragma acc data present(A[0:n*n], A_new[0:n*n])
//Начало параллельной области с указанием асинхронности выполнения
#pragma acc kernels async(1)
            {
//Распараллеливанием циклы по итерациям и потокам
#pragma acc loop independent collapse(2)
                for (int i = 1; i < n - 1; i++) {
                    for (int j = 1; j < n - 1; j++) {
                        A_new[i * n + j] =
                                0.25 * (A[(i + 1) * n + j] + A[(i - 1) * n + j] + A[i * n + j - 1] + A[i * n + j + 1]);
                    }
                }
            }
            int id = 0;
//Ожидание завершения асинхронных операций
#pragma acc wait
//Копируем данные с устройства на хост
#pragma acc host_data use_device(A, A_new)
            {
                cublasDaxpy(handle, n * n, &alpha, A_new, 1, A, 1);
                cublasIdamax(handle, n * n, A, 1, &id);

            }
//Копируем значения элемента с максимальным значением на хост
#pragma acc update self(u[id-1:1])
            error = fabs(A[id - 1]);
//Копируем данные с хоста на устройство
#pragma acc host_data use_device(A, A_new)
            cublasDcopy(handle, n * n, A_new, 1, A, 1);

        } else {
//Создаём копии массивов A и A_new на устройстве
#pragma acc data present(A[0:n*n], A_new[0:n*n])
//Указываем начало параллельной области с указанием асинхронности выполнения
#pragma acc kernels async(1)
            {
//Распараллеливаем циклы по итерациям и потокам
#pragma acc loop independent collapse(2)
                for (int i = 1; i < n - 1; i++) {
                    for (int j = 1; j < n - 1; j++) {
                        A_new[i * n + j] =
                                0.25 * (A[(i + 1) * n + j] + A[(i - 1) * n + j] + A[i * n + j - 1] + A[i * n + j + 1]);
                    }
                }
            }
        }
        tmp = A;
        A = A_new;
        A_new = tmp;
        
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
