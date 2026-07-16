struct FftParams {
    unsigned int n;
    unsigned int stage;
    unsigned int inverse;
    unsigned int batch_count;
};

__device__ unsigned int fft_bit_reverse(unsigned int value, unsigned int bits) {
    unsigned int reversed = 0U;
    for (unsigned int remaining = bits; remaining != 0U; --remaining) {
        reversed = (reversed << 1U) | (value & 1U);
        value >>= 1U;
    }
    return reversed;
}

__device__ unsigned int fft_log2(unsigned int value) {
    unsigned int bits = 0U;
    for (value >>= 1U; value != 0U; value >>= 1U) {
        ++bits;
    }
    return bits;
}

extern "C" __global__ void fft_bitrev(
    float* data_re,
    float* data_im,
    FftParams params
) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int total = params.n * params.batch_count;
    if (index >= total) {
        return;
    }

    const unsigned int row = index / params.n;
    const unsigned int local_index = index % params.n;
    const unsigned int reversed = fft_bit_reverse(local_index, fft_log2(params.n));
    if (reversed > local_index) {
        const unsigned int base = row * params.n;
        const unsigned int left = base + local_index;
        const unsigned int right = base + reversed;
        const float real = data_re[left];
        const float imaginary = data_im[left];
        data_re[left] = data_re[right];
        data_im[left] = data_im[right];
        data_re[right] = real;
        data_im[right] = imaginary;
    }
}

extern "C" __global__ void fft_forward(
    float* data_re,
    float* data_im,
    FftParams params
) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int half_n = params.n >> 1U;
    const unsigned int total = half_n * params.batch_count;
    if (index >= total) {
        return;
    }

    const unsigned int row = index / half_n;
    const unsigned int local_index = index % half_n;
    const unsigned int half_group = 1U << params.stage;
    const unsigned int group_size = half_group << 1U;
    const unsigned int group = local_index / half_group;
    const unsigned int offset = local_index % half_group;
    const unsigned int base = row * params.n;
    const unsigned int even = base + group * group_size + offset;
    const unsigned int odd = even + half_group;

    float angle = -6.28318530717958647692F * static_cast<float>(offset)
        / static_cast<float>(group_size);
    if (params.inverse != 0U) {
        angle = -angle;
    }
    const float weight_re = cosf(angle);
    const float weight_im = sinf(angle);
    const float even_re = data_re[even];
    const float even_im = data_im[even];
    const float odd_re = data_re[odd];
    const float odd_im = data_im[odd];
    const float product_re = weight_re * odd_re - weight_im * odd_im;
    const float product_im = weight_re * odd_im + weight_im * odd_re;

    data_re[even] = even_re + product_re;
    data_im[even] = even_im + product_im;
    data_re[odd] = even_re - product_re;
    data_im[odd] = even_im - product_im;
}

extern "C" __global__ void fft_scale(
    float* data_re,
    float* data_im,
    FftParams params
) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int total = params.n * params.batch_count;
    if (index >= total) {
        return;
    }
    const float inverse_length = 1.0F / static_cast<float>(params.n);
    data_re[index] *= inverse_length;
    data_im[index] *= inverse_length;
}
