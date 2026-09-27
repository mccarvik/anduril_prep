def reverseBits(n: int) -> int:
    result = 0
    for _ in range(32):
        result = (result << 1) | (n & 1)  # take lowest bit of n, append to result
        n >>= 1
    return result