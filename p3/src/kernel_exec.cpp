bool solveDevice(const cl_context cxContext, const cl_command_queue cqCommandQueue, const cl_program cpProgram, const cl_mem clChanges, cl_mem clAccount, cl_mem clSum, const int clients, const int periods) {

    //TODO you may need to change this code, e.g., execute more kernels, or
    // change the parallelism

    // create a kernel
    cl_int ciErr;
    cl_kernel ckKernel1 = clCreateKernel(cpProgram, "compute", &ciErr);
    if (!ckKernel1 || ciErr != CL_SUCCESS)
    {
        printf("Error: Failed to create compute kernel!\n");
        return false;
    }

    // set kernel parameters
    ciErr = clSetKernelArg(ckKernel1, 0, sizeof(cl_mem), &clChanges);
    ciErr |= clSetKernelArg(ckKernel1, 1, sizeof(cl_mem), &clAccount);
    ciErr |= clSetKernelArg(ckKernel1, 2, sizeof(cl_mem), &clSum);
    ciErr |= clSetKernelArg(ckKernel1, 3, sizeof(int), &clients);
    ciErr |= clSetKernelArg(ckKernel1, 4, sizeof(int), &periods);
    if (ciErr != CL_SUCCESS) {
       fprintf(stderr, "Error %i in clSetKernelArg.\n", ciErr);
       return NULL;
    }

    // configure parallelism
    size_t global[1], local[1];
    global[0] = clients;
    local[0] = 128;

    // enqueue kernel
    ciErr = clEnqueueNDRangeKernel(cqCommandQueue, ckKernel1, 1, NULL, global, local, 0, NULL, NULL);
    if (ciErr != CL_SUCCESS) {
        fprintf(stderr, "Error %i in clEnqueueNDRangeKernel.\n", ciErr);
        return NULL;
    }

    clReleaseKernel(ckKernel1);

    return true;
}
