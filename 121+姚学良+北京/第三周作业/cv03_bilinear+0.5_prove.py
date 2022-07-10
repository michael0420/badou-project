'''
    The src image is not same size equal to dst image,
    According to the equations:
        srcX = dstX * (srcW / dstW)
        srcY = dstY * (srcH / dstH)
    The central point equation:
        src(centre) = src(M - 1)/2  presume the src image is size of M * M
        dst(centre) = dst(N - 1)/2  presume the dst image is size of N * N
    Assume thers is a Z that lets equations go to:
        srcX + Z = (dstX + Z) * (srcW / dstW)
        srcY + Z = (dstY + Z) * (srcH / dstH)
    To make the central point of both images match:
        (M - 1) / 2 + Z = ((N - 1) / 2 + Z) * (M / N)
        (M - 1) / 2 + Z = (N - 1) / 2  * (M / N) + Z * (M / N)
        Z * ((N - M) / N) = ((N - 1) / 2) * (M / N) - (M - 1) / 2
        Z * ((N - M) / N) = (N - M) / 2N
        Z = 1 / 2

'''