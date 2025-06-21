#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <CL/opencl.h>

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "kernel_C.cpp"
#include "kernel_exec.cpp"

#define CLIENTS 8192
#define PERIODS 8192

void startTimer(timeval *timer){
    gettimeofday(timer, NULL);
}

int stopTimer(timeval *timer){
    timeval tmp;
    gettimeofday(&tmp, NULL);
    tmp.tv_sec -= timer->tv_sec;
    tmp.tv_usec -= timer->tv_usec;
    if (tmp.tv_usec < 0){
        tmp.tv_usec+=1000000;
        tmp.tv_sec--;
    }
    return int(tmp.tv_usec + tmp.tv_sec*1000000);
}

static char *load_program_source(const char *filename)
{
    struct stat statbuf;
    FILE   *fh;
    char   *source;
    size_t fsize;

    fh = fopen(filename, "r");
    if (fh == 0)
        return 0;

    stat(filename, &statbuf);
    fsize = statbuf.st_size;
    source = (char *) malloc(fsize + 1);
    fread(source, fsize, 1, fh);
    source[fsize] = '\0';

    return source;
}

void printDeviceInfo(cl_device_id device_id) {
    char device_string[1024];
    clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(device_string), &device_string, NULL);
    printf(" CL_DEVICE_NAME: %s\n", device_string);
    clGetDeviceInfo(device_id, CL_DEVICE_VENDOR, sizeof(device_string), &device_string, NULL);
    printf(" CL_DEVICE_VENDOR: %s\n", device_string);
}

int main(int argc, char **argv) {
    int* changes = NULL;	// input account changes
    int* account = NULL;        // output account balance
    int* sum = NULL;            // output sum per period
    cl_mem clChanges;		// device copy of changes
    cl_mem clAccount;           // device copy of account
    cl_mem clSum;               // device copy of sum
    int* accountTest = NULL;    // account computed by accelerated code
    int* sumTest = NULL;        // sum computed vy accelerated code

    // parse command line
    int platformIndex = 0;
    int deviceIndex = 0;
    if (argc >= 2) 
        platformIndex = atoi(argv[1]);
    if (argc >= 3)
        deviceIndex = atoi(argv[2]);

    // get platforms (max 5)
    cl_int ciErr;
    cl_uint platforms = 0;
    cl_platform_id cpPlatform[5];
    ciErr = clGetPlatformIDs(5, cpPlatform, &platforms);
    if (ciErr != CL_SUCCESS) {
        fprintf(stderr, "Error %i in clGetPlatformID.\n", ciErr);
        return 1;
    }
    if (platformIndex > platforms-1) {
        fprintf(stderr, "Error: platform %i does not exist!\n.", platformIndex);
        return 1;
    }
    
    // get devices
    cl_uint devices = 0;
    cl_device_id cdDevices[10];
    ciErr = clGetDeviceIDs(cpPlatform[platformIndex], CL_DEVICE_TYPE_ALL, 10, cdDevices, &devices);
    if (ciErr != CL_SUCCESS) {
        fprintf(stderr, "Error %i in clGetDeviceIDs.\n", ciErr);
        return 1;
    }
    if (deviceIndex > devices-1) {
        fprintf(stderr, "Error: device %i does not exist!\n.", deviceIndex);
        return 1;
    }
    printDeviceInfo(cdDevices[deviceIndex]);
    
    // create context
    cl_context cxContext;
    cxContext = clCreateContext(0, 1, &(cdDevices[deviceIndex]), NULL, NULL, &ciErr);
    if (ciErr != CL_SUCCESS)
    {
        fprintf(stderr, "Error %i in clCreateContext.\n", ciErr);
        return 1;
    }

    // create queue
    cl_command_queue cqCommandQueue;
    cqCommandQueue = clCreateCommandQueue(cxContext, cdDevices[deviceIndex], NULL, &ciErr);
    if (ciErr != CL_SUCCESS)
    {
        fprintf(stderr, "Error %i in clCreateCommandQueue.\n", ciErr);
        return 1;
    }

    // load OpenCL program
    char *cSource = load_program_source("kernel.cl");
    if(!cSource) {
        fprintf(stderr, "Error: Failed to load OpenCL program.\n");
        return 1;
    }

    // create program
    cl_program cpProgram;
    cpProgram = clCreateProgramWithSource(cxContext, 1, (const char **)&cSource, NULL, &ciErr);
    if (ciErr != CL_SUCCESS)
    {
        fprintf(stderr, "Error %i in clCreateProgramWithSource.\n", ciErr);
        return 1;
    }

    // build the program
    ciErr = clBuildProgram(cpProgram, 0, NULL, NULL, NULL, NULL);
    if (ciErr != CL_SUCCESS)
    {
        fprintf(stderr, "Error in clBuildProgram (%i).\n", ciErr);
        char *build_log;
        size_t ret_val_size;
        clGetProgramBuildInfo(cpProgram, cdDevices[deviceIndex], CL_PROGRAM_BUILD_LOG, 0, NULL, &ret_val_size);
        build_log = (char*) malloc((ret_val_size+1)*sizeof(*build_log));
        clGetProgramBuildInfo(cpProgram, cdDevices[deviceIndex], CL_PROGRAM_BUILD_LOG, ret_val_size, build_log, NULL);
        build_log[ret_val_size] = '\0';
        fprintf(stderr, "%s\n", build_log);
        free(build_log);
        return 1;
    }

    // allocate and set host memory
    changes = (int*)malloc(CLIENTS*PERIODS*sizeof(changes[0]));
    account = (int*)malloc(CLIENTS*PERIODS*sizeof(account[0]));
    sum = (int*)calloc(PERIODS, sizeof(sum[0])); //XXX zeroize
    accountTest = (int*)malloc(CLIENTS*PERIODS*sizeof(accountTest[0]));
    sumTest = (int*)malloc(PERIODS*sizeof(sumTest[0]));

    // create random input
    for (int i = 0; i < CLIENTS*PERIODS; i++)
        changes[i] = int(100.0f*(float)rand() / float(RAND_MAX));

    // allocate the OpenCL buffer memory objects
    clChanges = clCreateBuffer(cxContext, CL_MEM_READ_ONLY, sizeof(cl_int) * CLIENTS*PERIODS, NULL, &ciErr);
    if (ciErr != CL_SUCCESS) {
        fprintf(stderr, "Error %i in clCreateBuffer.\n", ciErr);
        return 1;
    }
    clAccount = clCreateBuffer(cxContext, CL_MEM_READ_WRITE, sizeof(cl_int) * CLIENTS*PERIODS, NULL, &ciErr);
    if (ciErr != CL_SUCCESS) {
        fprintf(stderr, "Error %i in clCreateBuffer.\n", ciErr);
        return 1;
    }
    clSum = clCreateBuffer(cxContext, CL_MEM_READ_WRITE, sizeof(cl_int) * PERIODS, NULL, &ciErr);
    if (ciErr != CL_SUCCESS) {
        fprintf(stderr, "Error %i in clCreateBuffer.\n", ciErr);
        return 1;
    }

    // copy input to device
    ciErr = clEnqueueWriteBuffer(cqCommandQueue, clChanges, true, 0, sizeof(int)*CLIENTS*PERIODS, changes, 0, NULL, NULL);
    //ciErr |= clEnqueueWriteBuffer(cqCommandQueue, clAccount, true, 0, sizeof(int)*CLIENTS*PERIODS, account, 0, NULL, NULL);
    ciErr |= clEnqueueWriteBuffer(cqCommandQueue, clSum, true, 0, sizeof(int)*PERIODS, sum, 0, NULL, NULL); // copy to zeroize device buffer
    if (ciErr != CL_SUCCESS) {
        fprintf(stderr, "Error %i in clEnqueueWriteBuffer.\n", ciErr);
        return 1;
    }
    clFinish(cqCommandQueue);

    // solve on CPU
    printf("Solving on CPU...\n");
    timeval timer;
    startTimer(&timer);
    solveCPU(changes, account, sum, CLIENTS, PERIODS);
    int time = stopTimer(&timer);
    printf("Performance: %f megavalues/s\n", (float)CLIENTS*(float)PERIODS/(float)time);

    // solve on device
    printf("Solving on device...\n");
    startTimer(&timer);
    if (!solveDevice(cxContext, cqCommandQueue, cpProgram, clChanges, clAccount, clSum, CLIENTS, PERIODS)) {
        fprintf(stderr, "Error in solveDevice.\n");
        return 1;
    }
    clFinish(cqCommandQueue);
    time = stopTimer(&timer);
    printf("Performance: %f megavalues/s\n", (float)CLIENTS*(float)PERIODS/(float)time);

    printf("Checking output...\n");

    // get device output
    ciErr = clEnqueueReadBuffer(cqCommandQueue, clAccount, true, 0, sizeof(int)*CLIENTS*PERIODS, accountTest, 0, NULL, NULL );
    ciErr |= clEnqueueReadBuffer(cqCommandQueue, clSum, true, 0, sizeof(int)*PERIODS, sumTest, 0, NULL, NULL );
    if (ciErr != CL_SUCCESS) {
        fprintf(stderr, "Error %i in clEnqueueReadBuffer.\n", ciErr);
        return 1;
    }

    // release OpenCL objects
    clReleaseMemObject(clChanges);
    clReleaseMemObject(clAccount);
    clReleaseMemObject(clSum);
    //clReleaseKernel(ckKernel);
    clReleaseProgram(cpProgram);
    clReleaseCommandQueue(cqCommandQueue);
    clReleaseContext(cxContext);

    // check results in account
    for (int j = 0; j < PERIODS; j++)
        for (int i = 0; i < CLIENTS; i++)
            if (account[j*CLIENTS+i] != accountTest[j*CLIENTS+i]) {
                fprintf(stderr, "Account data mismatch at index %i, %i: %i should be %i :-(\n", i, j, accountTest[j*CLIENTS+i], account[j*CLIENTS+i]);
                return 1;
            }

    // check results in sum
    for (int i = 0; i < PERIODS; i++)
        if (sum[i] != sumTest[i]) {
            fprintf(stderr, "Sum data mismatch at index %i: %i should be %i :-(\n", i, sumTest[i], sum[i]);
            return 1;
        }
    printf("Test OK.\n");

    return 0;
}

