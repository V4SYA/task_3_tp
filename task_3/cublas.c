/*Данный код решает уравнение Пуассона на прямоугольной области с помощью метода Якоби и ускоряет вычисления с помощью директив OpenACC.
Вначале определяются параметры модели, такие как точность, размер сетки и максимальное число итераций.
Затем выделяется память на хосте и девайсе для двух массивов A и Anew,
которые хранят значения температуры на сетке.

Основной алгоритм выполняется до тех пор, пока значение переменной error не станет меньше заданного порога.
На каждой итерации цикла происходит вычисление новых значений на текущей точке с использованием значений
соседних точек и обновление значения error. Когда значение error становится меньше заданного порога,
цикл прерывается и программа выводит результаты решения уравнения Пуассона на экран*/

#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

char ERROR_WITH_ARGS[] = ">>> Not enough args\n";
char ERROR_WITH_ARG_1[] = ">>> Incorrect first param\n";
char ERROR_WITH_ARG_2[] = ">>> Incorrect second param\n";
char ERROR_WITH_ARG_3[] = ">>> Incorrect third param\n";

double CORNER_1 = 10;
double CORNER_2 = 20;
double CORNER_3 = 30;
double CORNER_4 = 20;

int main(int argc, char *argv[]) {
    int max_num_iter, n;
    double max_acc;
    if (argc < 4){
        printf(ERROR_WITH_ARGS);
        exit(1);
    } else{
        n = atoi(argv[1]); // Размер сетки
        if (n == 0){
            printf(ERROR_WITH_ARG_1);
            exit(1);
        }
        max_num_iter = atoi(argv[2]); // Количество итераций
        if (max_num_iter == 0){
            printf(ERROR_WITH_ARG_2);
            exit(1);
        }
        max_acc = atof(argv[3]); // Точность
        if (max_acc == 0){
            printf(ERROR_WITH_ARG_3);
            exit(1);
        }
    }
   
    clock_t a = clock();

    double* A = (double*)calloc(n * n, sizeof(double));
    double* Anew = (double*)calloc(n * n, sizeof(double));

    A[0] = CORNER_1;
    A[n-1] = CORNER_2;
    A[n * n - 1] = CORNER_3;
    A[n * (n - 1)] = CORNER_4;

    int num_iter = 0;
    double error = 1 + max_acc;
    double step = (10.0 / (n - 1));
    
// выделение памяти на устройстве и копирование данных из памяти хоста в память устройства
#pragma acc enter data create(A[0:n*n], Anew[0:n*n]) copyin(n, step)
    
// ядро, выполняющее циклическое заполнение массива A
#pragma acc kernels
    {
#pragma acc loop independent
        for (int i = 1; i < n - 1; i++) {
            A[i] = CORNER_1 + i * step;
            A[i * n] = CORNER_1 + i * step;
            A[i * n + (n - 1)] = CORNER_2 + i * step;
            A[n * (n - 1) + i] = CORNER_4 + i * step;
        }
    }
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    double *tmp;

    while (max_num_iter > num_iter && max_acc < error) {
        num_iter++;
        if (num_iter % 5 == 0) {
            
//объявляется область данных, которые находятся на устройстве
#pragma acc data present(A[0:n*n], Anew[0:n*n])
            
//async(1) указывает, что выполнение этого блока должно начаться после выполнения предыдущего блока
#pragma acc kernels async(1)
            {
//директива указывает на то, что следующий цикл for может быть распараллелен, collapse(2) отвечает за то, что оба вложенных цикла могут быть распараллелены
#pragma acc loop independent collapse(2)
                for (int i = 1; i < n - 1; i++) {
                    for (int j = 1; j < n - 1; j++) {
                        A[i * n + j] = 0.25 * (Anew[(i + 1) * n + j] + Anew[(i - 1) * n + j] + Anew[i * n + j - 1] + Anew[i * n + j + 1]);
                    }
                }
            }
           
            int max_id = 0; //хранение индекса максимального элемента массива Anew
            const double alpha = -1;
            
// останавливает выполнение программы, пока не завершатся все ядра, запущенные с использованием async()
#pragma acc wait
            
//директива определяет, что данные массивов находятся и на устройстве, и на хосте, и могут использоваться и изменяться на обоих уровнях
#pragma acc host_data use_device(A, Anew)
            {
                cublasDaxpy(handle, n * n, &alpha, A, 1, Anew, 1); // функция вычисляет значение -1 * A + Anew и сохраняет результат в Anew
                cublasIdamax(handle, n * n, Anew, 1, &max_id); //находит индекс максимального элемента массива Anew
            }
//копирует один элемент массива Anew с индексом max_id-1 с устройства на хост
#pragma acc update self(Anew[max_id-1:1])
            
            error = fabs(Anew[max_id - 1]);
#pragma acc host_data use_device(A, Anew)
            cublasDcopy(handle, n * n, A, 1, Anew, 1); //функция копирует содержимое массива 
            
//указывает, что все ранее запланированные ядра и данные, связанные с ускорителем, должны завершить свою работу, прежде чем продолжить выполнение кода на хост-процессоре
#pragma acc wait(1)
            printf("Номер итерации: %d, ошибка: %0.8lf\n", num_iter, error);
            

        }
        else {
//указываю, что данные A и Anew должны быть доступны на ускорителе во время выполнения цикла
#pragma acc data present(A[0:n*n], Anew[0:n*n])
            
//запускаю параллельное выполнение цикла на ускорителе с помощью ядер
//async(1) указывает, что этот цикл должен выполняться асинхронно с остальной программой
#pragma acc kernels async(1)
            {
//указываю, что цикл должен быть распараллелен на ускорителе
// collapse(2) указывает, что два вложенных цикла могут быть объединены в один для более эффективного распараллеливания
#pragma acc loop independent collapse(2)
                for (int i = 1; i < n - 1; i++) {
                    for (int j = 1; j < n - 1; j++) {
                        A[i * n + j] = 0.25 * (Anew[(i + 1) * n + j] + 
                        Anew[(i - 1) * n + j] + 
                        Anew[i * n + j - 1] + 
                        Anew[i * n + j + 1]);
                    }
                }
            }
        }

        tmp = Anew;
        Anew = A;
        A = tmp;
       
    }

    printf("Result: %d, %0.6lf\n", num_iter, error);
    clock_t b = clock();
    double d = (double)(b-a)/CLOCKS_PER_SEC; // перевожу в секунды 
    printf("Time: %.25f ", d);

    cublasDestroy(handle); //освобождаю ресурсы, связанные с объектом handle
    free(A); 
    free(Anew);
    return 0;
}
