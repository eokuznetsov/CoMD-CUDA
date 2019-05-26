/*************************************************************************
 * Copyright (c) 2013, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 ************************************************************************/

#ifndef __GPU_SCAN_H_
#define __GPU_SCAN_H_

#include <hip/hip_runtime.h>
#ifdef AMD_PLATFORM
#include <hipcub/hipcub.hpp>
#else
#include <cub/cub.cuh>
#endif

void scan(int *data, int n, int *partial_sums, hipStream_t stream)
{
  size_t temp_size = n+1;
  if (temp_size % 256 != 0) temp_size = ((temp_size + 255)/256)*256;	// pad to 256 elements
  temp_size *= sizeof(int);

    //int* temp_data;
    //hipMalloc(&temp_data, n* sizeof(int));
    //hipMemcpy(temp_data, data, sizeof(int) * n, hipMemcpyDeviceToDevice);
    //hipcub::DeviceScan::ExclusiveSum(partial_sums, temp_size, temp_data, data, n, stream);
    ////sum_scan_blelloch((unsigned int*)partial_sums, (unsigned int*)data, (size_t)n, stream);
    //hipFree(temp_data);

    //int* temp_data = new int[n];
    //hipMemcpy(temp_data, data, sizeof(int) * n, hipMemcpyDeviceToHost);
    //int sum = 0;
    //for(int i=0;i<n;++i){
    //    int curr = temp_data[i];
    //    temp_data[i] = sum;
    //    sum += curr;
    //}
    //hipMemcpy(data, temp_data, sizeof(int) * n, hipMemcpyHostToDevice);
    //delete[] temp_data;
#ifdef AMD_PLATFORM
#ifdef DEBUG
  size_t   temp_storage_bytes = 0;
  hipcub::DeviceScan::ExclusiveSum(nullptr, temp_storage_bytes, data, data, n, stream);
  assert(temp_storage_bytes <= temp_size);
#endif
  hipcub::DeviceScan::ExclusiveSum(partial_sums, temp_size, data, data, n, stream);
#else
  cub::DeviceScan::ExclusiveSum(partial_sums, temp_size, data, data, n, stream);
#endif
}

#endif
