import sympy as sp

def quaternion_to_DCM(w, x, y, z):
    C = sp.Matrix([
        [w**2 + x**2 - y**2 - z**2, 2*(x*y + w*z), 2*(x*z - w*y)],
        [2*(x*y - w*z), w**2 - x**2 + y**2 - z**2, 2*(y*z + w*x)],
        [2*(x*z + w*y), 2*(y*z - w*x), w**2 - x**2 - y**2 + z**2]])

    return C

if __name__=="__main__":
    b0, b1, b2, b3, bp0, bp1, bp2, bp3, bpp0, bpp1, bpp2, bpp3 = \
        sp.symbols("b0 b1 b2 b3 bp0, bp1, bp2, bp3, bpp0, bpp1, bpp2, bpp3")
    C_FB = quaternion_to_DCM(bpp0, bpp1, bpp2, bpp3)
    C_BN = quaternion_to_DCM(bp0, bp1, bp2, bp3)

    C_FN = C_FB @ C_BN

    b0 = 0.5*sp.sqrt(sp.trace(C_FN) + 1)
    pattern1 = bp0**2 + bp1**2 + bp2**2 + bp3**2
    pattern2 = bpp0**2 + bpp1**2 + bpp2**2 + bpp3**2
    # b0 = b0.replace(pattern1, 1)
    # b0 = b0.replace(pattern2, 1)
    # print(b0)
    C_FN = C_FN.replace(pattern1, 1)
    C_FN = C_FN.replace(pattern2, 1)
    print(C_FN)