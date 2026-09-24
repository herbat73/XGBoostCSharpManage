// Managed ports of the fdlibm (FreeBSD msun) special functions that .NET does not provide:
// erf, log1p, log1pf, expm1 and lgammaf. Constants and branch structure follow the originals.
//
// ====================================================
// Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
//
// Developed at SunPro, a Sun Microsystems, Inc. business.
// Permission to use, copy, modify, and distribute this
// software is freely granted, provided that this notice
// is preserved.
// ====================================================
using System.Runtime.CompilerServices;

namespace XGBoost.Common;

internal static class FdLibm
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int HighWord(double x) => (int)(BitConverter.DoubleToInt64Bits(x) >> 32);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static double SetHighWord(double x, int hi) =>
        BitConverter.Int64BitsToDouble(((long)hi << 32) | (BitConverter.DoubleToInt64Bits(x) & 0xffffffffL));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static double SetLowWord(double x, uint lo) =>
        BitConverter.Int64BitsToDouble((BitConverter.DoubleToInt64Bits(x) & unchecked((long)0xffffffff00000000UL)) | lo);

    // ---- erf ---------------------------------------------------------------------------------
    private const double Tiny = 1e-300;
    private const double Erx = 8.45062911510467529297e-01;
    private const double Efx = 1.28379167095512586316e-01;
    private const double Efx8 = 1.02703333676410069053e+00;
    private const double Pp0 = 1.28379167095512558561e-01, Pp1 = -3.25042107247001499370e-01, Pp2 = -2.84817495755985104766e-02,
        Pp3 = -5.77027029648944159157e-03, Pp4 = -2.37630166566501626084e-05;
    private const double Qq1 = 3.97917223959155352819e-01, Qq2 = 6.50222499887672944485e-02, Qq3 = 5.08130628187576562776e-03,
        Qq4 = 1.32494738004321644526e-04, Qq5 = -3.96022827877536812320e-06;
    private const double Pa0 = -2.36211856075265944077e-03, Pa1 = 4.14856118683748331666e-01, Pa2 = -3.72207876035701323847e-01,
        Pa3 = 3.18346619901161753674e-01, Pa4 = -1.10894694282396677476e-01, Pa5 = 3.54783043256182359371e-02,
        Pa6 = -2.16637559486879084300e-03;
    private const double Qa1 = 1.06420880400844228286e-01, Qa2 = 5.40397917702171048937e-01, Qa3 = 7.18286544141962662868e-02,
        Qa4 = 1.26171219808761642112e-01, Qa5 = 1.36370839120290507362e-02, Qa6 = 1.19844998467991074170e-02;
    private const double Ra0 = -9.86494403484714822705e-03, Ra1 = -6.93858572707181764372e-01, Ra2 = -1.05586262253232909814e+01,
        Ra3 = -6.23753324503260060396e+01, Ra4 = -1.62396669462573470355e+02, Ra5 = -1.84605092906711035994e+02,
        Ra6 = -8.12874355063065934246e+01, Ra7 = -9.81432934416914548592e+00;
    private const double Sa1 = 1.96512716674392571292e+01, Sa2 = 1.37657754143519042600e+02, Sa3 = 4.34565877475229228821e+02,
        Sa4 = 6.45387271733267880336e+02, Sa5 = 4.29008140027567833386e+02, Sa6 = 1.08635005541779435134e+02,
        Sa7 = 6.57024977031928170135e+00, Sa8 = -6.04244152148580987438e-02;
    private const double Rb0 = -9.86494292470009928597e-03, Rb1 = -7.99283237680523006574e-01, Rb2 = -1.77579549177547519889e+01,
        Rb3 = -1.60636384855821916062e+02, Rb4 = -6.37566443368389627722e+02, Rb5 = -1.02509513161107724954e+03,
        Rb6 = -4.83519191608651397019e+02;
    private const double Sb1 = 3.03380607434824582924e+01, Sb2 = 3.25792512996573918826e+02, Sb3 = 1.53672958608443695994e+03,
        Sb4 = 3.19985821950859553908e+03, Sb5 = 2.55305040643316442583e+03, Sb6 = 4.74528541206955367215e+02,
        Sb7 = -2.24409524465858183362e+01;

    public static double Erf(double x)
    {
        var hx = HighWord(x);
        var ix = hx & 0x7fffffff;
        if (ix >= 0x7ff00000)
        {
            var i = (int)(((uint)hx >> 31) << 1);
            return (1 - i) + 1.0 / x;
        }

        double r, s, y, z;
        if (ix < 0x3feb0000)
        {
            if (ix < 0x3e300000)
            {
                if (ix < 0x00800000) return (8 * x + Efx8 * x) / 8;
                return x + Efx * x;
            }
            z = x * x;
            r = Pp0 + z * (Pp1 + z * (Pp2 + z * (Pp3 + z * Pp4)));
            s = 1.0 + z * (Qq1 + z * (Qq2 + z * (Qq3 + z * (Qq4 + z * Qq5))));
            y = r / s;
            return x + x * y;
        }
        if (ix < 0x3ff40000)
        {
            s = Math.Abs(x) - 1.0;
            var p = Pa0 + s * (Pa1 + s * (Pa2 + s * (Pa3 + s * (Pa4 + s * (Pa5 + s * Pa6)))));
            var q = 1.0 + s * (Qa1 + s * (Qa2 + s * (Qa3 + s * (Qa4 + s * (Qa5 + s * Qa6)))));
            return hx >= 0 ? Erx + p / q : -Erx - p / q;
        }
        if (ix >= 0x40180000) return hx >= 0 ? 1.0 - Tiny : Tiny - 1.0;
        x = Math.Abs(x);
        s = 1.0 / (x * x);
        double rr, ss;
        if (ix < 0x4006DB6E)
        {
            rr = Ra0 + s * (Ra1 + s * (Ra2 + s * (Ra3 + s * (Ra4 + s * (Ra5 + s * (Ra6 + s * Ra7))))));
            ss = 1.0 + s * (Sa1 + s * (Sa2 + s * (Sa3 + s * (Sa4 + s * (Sa5 + s * (Sa6 + s * (Sa7 + s * Sa8)))))));
        }
        else
        {
            rr = Rb0 + s * (Rb1 + s * (Rb2 + s * (Rb3 + s * (Rb4 + s * (Rb5 + s * Rb6)))));
            ss = 1.0 + s * (Sb1 + s * (Sb2 + s * (Sb3 + s * (Sb4 + s * (Sb5 + s * (Sb6 + s * Sb7))))));
        }
        z = SetLowWord(x, 0);
        r = Math.Exp(-z * z - 0.5625) * Math.Exp((z - x) * (z + x) + rr / ss);
        return hx >= 0 ? 1.0 - r / x : r / x - 1.0;
    }

    // ---- log1p -------------------------------------------------------------------------------
    private const double Ln2Hi = 6.93147180369123816490e-01, Ln2Lo = 1.90821492927058770002e-10, Two54 = 1.80143985094819840000e+16;
    private const double Lp1 = 6.666666666666735130e-01, Lp2 = 3.999999999940941908e-01, Lp3 = 2.857142874366239149e-01,
        Lp4 = 2.222219843214978396e-01, Lp5 = 1.818357216161805012e-01, Lp6 = 1.531383769920937332e-01,
        Lp7 = 1.479819860511658591e-01;

    public static double Log1P(double x)
    {
        double hfsq, f = 0, c = 0, s, z, R, u;
        int k, hu = 0;
        var hx = HighWord(x);
        var ax = hx & 0x7fffffff;

        k = 1;
        if (hx < 0x3FDA827A)
        {
            if (ax >= 0x3ff00000)
            {
                if (x == -1.0) return double.NegativeInfinity;
                return double.NaN;
            }
            if (ax < 0x3e200000)
            {
                if (Two54 + x > 0 && ax < 0x3c900000) return x;
                return x - x * x * 0.5;
            }
            if (hx > 0 || hx <= unchecked((int)0xbfd2bec4))
            {
                k = 0;
                f = x;
                hu = 1;
            }
        }
        if (hx >= 0x7ff00000) return x + x;
        if (k != 0)
        {
            if (hx < 0x43400000)
            {
                u = 1.0 + x;
                hu = HighWord(u);
                k = (hu >> 20) - 1023;
                c = k > 0 ? 1.0 - (u - x) : x - (u - 1.0);
                c /= u;
            }
            else
            {
                u = x;
                hu = HighWord(u);
                k = (hu >> 20) - 1023;
                c = 0;
            }
            hu &= 0x000fffff;
            if (hu < 0x6a09e)
            {
                u = SetHighWord(u, hu | 0x3ff00000);
            }
            else
            {
                k += 1;
                u = SetHighWord(u, hu | 0x3fe00000);
                hu = (0x00100000 - hu) >> 2;
            }
            f = u - 1.0;
        }
        hfsq = 0.5 * f * f;
        if (hu == 0)
        {
            if (f == 0)
            {
                if (k == 0) return 0;
                c += k * Ln2Lo;
                return k * Ln2Hi + c;
            }
            R = hfsq * (1.0 - 0.66666666666666666 * f);
            if (k == 0) return f - R;
            return k * Ln2Hi - ((R - (k * Ln2Lo + c)) - f);
        }
        s = f / (2.0 + f);
        z = s * s;
        R = z * (Lp1 + z * (Lp2 + z * (Lp3 + z * (Lp4 + z * (Lp5 + z * (Lp6 + z * Lp7))))));
        if (k == 0) return f - (hfsq - s * (hfsq + R));
        return k * Ln2Hi - ((hfsq - (s * (hfsq + R) + (k * Ln2Lo + c))) - f);
    }

    // ---- log1pf ------------------------------------------------------------------------------
    private const float Ln2HiF = 6.9313812256e-01f, Ln2LoF = 9.0580006145e-06f, Two25F = 3.355443200e+07f;
    private const float Lp1F = 6.6666668653e-01f, Lp2F = 4.0000000596e-01f, Lp3F = 2.8571429849e-01f, Lp4F = 2.2222198546e-01f,
        Lp5F = 1.8183572590e-01f, Lp6F = 1.5313838422e-01f, Lp7F = 1.4798198640e-01f;

    public static float Log1PF(float x)
    {
        float hfsq, f = 0, c = 0, s, z, R, u;
        int k, hu = 0;
        var hx = BitConverter.SingleToInt32Bits(x);
        var ax = hx & 0x7fffffff;

        k = 1;
        if (hx < 0x3ed413d0)
        {
            if (ax >= 0x3f800000)
            {
                if (x == -1.0f) return float.NegativeInfinity;
                return float.NaN;
            }
            if (ax < 0x38000000)
            {
                if (Two25F + x > 0 && ax < 0x33800000) return x;
                return x - x * x * 0.5f;
            }
            if (hx > 0 || hx <= unchecked((int)0xbe95f619))
            {
                k = 0;
                f = x;
                hu = 1;
            }
        }
        if (hx >= 0x7f800000) return x + x;
        if (k != 0)
        {
            if (hx < 0x5a000000)
            {
                u = 1.0f + x;
                hu = BitConverter.SingleToInt32Bits(u);
                k = (hu >> 23) - 127;
                c = k > 0 ? 1.0f - (u - x) : x - (u - 1.0f);
                c /= u;
            }
            else
            {
                u = x;
                hu = BitConverter.SingleToInt32Bits(u);
                k = (hu >> 23) - 127;
                c = 0;
            }
            hu &= 0x007fffff;
            if (hu < 0x3504f4)
            {
                u = BitConverter.Int32BitsToSingle(hu | 0x3f800000);
            }
            else
            {
                k += 1;
                u = BitConverter.Int32BitsToSingle(hu | 0x3f000000);
                hu = (0x00800000 - hu) >> 2;
            }
            f = u - 1.0f;
        }
        hfsq = 0.5f * f * f;
        if (hu == 0)
        {
            if (f == 0)
            {
                if (k == 0) return 0;
                c += k * Ln2LoF;
                return k * Ln2HiF + c;
            }
            R = hfsq * (1.0f - 0.66666666666666666f * f);
            if (k == 0) return f - R;
            return k * Ln2HiF - ((R - (k * Ln2LoF + c)) - f);
        }
        s = f / (2.0f + f);
        z = s * s;
        R = z * (Lp1F + z * (Lp2F + z * (Lp3F + z * (Lp4F + z * (Lp5F + z * (Lp6F + z * Lp7F))))));
        if (k == 0) return f - (hfsq - s * (hfsq + R));
        return k * Ln2HiF - ((hfsq - (s * (hfsq + R) + (k * Ln2LoF + c))) - f);
    }

    // ---- expm1 -------------------------------------------------------------------------------
    private const double OThreshold = 7.09782712893383973096e+02, InvLn2 = 1.44269504088896338700e+00, Huge = 1.0e+300;
    private const double Q1 = -3.33333333333331316428e-02, Q2 = 1.58730158725481460165e-03, Q3 = -7.93650757867487942473e-05,
        Q4 = 4.00821782732936239552e-06, Q5 = -2.01099218183624371326e-07;

    public static double ExpM1(double x)
    {
        double y, hi, lo, c = 0, t, e, hxs, hfx, r1, twopk;
        int k;
        var hxw = (uint)HighWord(x);
        var xsb = (int)(hxw & 0x80000000);
        var hx = hxw & 0x7fffffff;

        if (hx >= 0x4043687A)
        {
            if (hx >= 0x40862E42)
            {
                if (hx >= 0x7ff00000)
                {
                    var low = (uint)BitConverter.DoubleToInt64Bits(x);
                    if (((hx & 0xfffff) | low) != 0) return x + x;
                    return xsb == 0 ? x : -1.0;
                }
                if (x > OThreshold) return double.PositiveInfinity;
            }
            if (xsb != 0)
            {
                if (x + Tiny < 0.0) return Tiny - 1.0;
            }
        }

        if (hx > 0x3fd62e42)
        {
            if (hx < 0x3FF0A2B2)
            {
                if (xsb == 0)
                {
                    hi = x - Ln2Hi;
                    lo = Ln2Lo;
                    k = 1;
                }
                else
                {
                    hi = x + Ln2Hi;
                    lo = -Ln2Lo;
                    k = -1;
                }
            }
            else
            {
                k = (int)(InvLn2 * x + (xsb == 0 ? 0.5 : -0.5));
                t = k;
                hi = x - t * Ln2Hi;
                lo = t * Ln2Lo;
            }
            x = hi - lo;
            c = (hi - x) - lo;
        }
        else if (hx < 0x3c900000)
        {
            t = Huge + x;
            return x - (t - (Huge + x));
        }
        else
        {
            k = 0;
        }

        hfx = 0.5 * x;
        hxs = x * hfx;
        r1 = 1.0 + hxs * (Q1 + hxs * (Q2 + hxs * (Q3 + hxs * (Q4 + hxs * Q5))));
        t = 3.0 - r1 * hfx;
        e = hxs * ((r1 - t) / (6.0 - x * t));
        if (k == 0) return x - (x * e - hxs);
        twopk = BitConverter.Int64BitsToDouble((long)((uint)(0x3ff + k) << 20) << 32);
        e = x * (e - c) - c;
        e -= hxs;
        if (k == -1) return 0.5 * (x - e) - 0.5;
        if (k == 1)
        {
            if (x < -0.25) return -2.0 * (e - (x + 0.5));
            return 1.0 + 2.0 * (x - e);
        }
        if (k <= -2 || k > 56)
        {
            y = 1.0 - (e - x);
            if (k == 1024) y = y * 2.0 * 8.98846567431157953865e+307;
            else y *= twopk;
            return y - 1.0;
        }
        t = 1.0;
        if (k < 20)
        {
            t = SetHighWord(t, 0x3ff00000 - (0x200000 >> k));
            y = t - (e - x);
            y *= twopk;
        }
        else
        {
            t = SetHighWord(t, (0x3ff - k) << 20);
            y = x - (e + t);
            y += 1.0;
            y *= twopk;
        }
        return y;
    }

    // ---- lgammaf -----------------------------------------------------------------------------
    private const float PiF = 3.1415927410e+00f;
    private const float A0 = 7.72156641e-02f, A1 = 3.22467119e-01f, A2 = 6.73484802e-02f, A3 = 2.06395667e-02f,
        A4 = 6.98275631e-03f, A5 = 4.11768444e-03f;
    private const float Tc = 1.46163213e+00f, Tf = -1.21486291e-01f, T0 = -2.94064460e-11f, T1 = -2.35939837e-08f,
        T2 = 4.83836412e-01f, T3 = -1.47586212e-01f, T4 = 6.46013096e-02f, T5 = -3.28450352e-02f, T6 = 1.86483748e-02f,
        T7 = -9.89206228e-03f;
    private const float U0 = -7.72156641e-02f, U1 = 7.36789703e-01f, U2 = 4.95649040e-01f, V1 = 1.10958421e+00f,
        V2 = 2.10598111e-01f, V3 = -1.02995494e-02f;
    private const float S0 = -7.72156641e-02f, S1 = 2.69987404e-01f, S2 = 1.42851010e-01f, S3 = 1.19389519e-02f,
        R1 = 6.79650068e-01f, R2 = 1.16058730e-01f, R3 = 3.75673687e-03f;
    private const float W0 = 4.18938547e-01f, W1 = 8.33332464e-02f, W2 = -2.76129087e-03f;

    public static float LGammaF(float x)
    {
        var hx = BitConverter.SingleToInt32Bits(x);
        var ix = hx & 0x7fffffff;
        if (ix >= 0x7f800000) return x * x;
        if (ix < 0x32000000)
        {
            if (ix == 0) return float.PositiveInfinity;
            return -MathF.Log(MathF.Abs(x));
        }

        var nadj = 0f;
        if (hx < 0)
        {
            if (ix >= 0x4b000000) return float.PositiveInfinity;
            // Reflection: sin(pi x) evaluated in double precision.
            var t = (float)Math.Sin(Math.PI * x);
            if (t == 0) return float.PositiveInfinity;
            nadj = MathF.Log(PiF / MathF.Abs(t * x));
            x = -x;
        }

        float r, y, z, p, p1, p2, q;
        int i;
        if (ix == 0x3f800000 || ix == 0x40000000)
        {
            r = 0;
        }
        else if (ix < 0x40000000)
        {
            if (ix <= 0x3f666666)
            {
                r = -MathF.Log(x);
                if (ix >= 0x3f3b4a20)
                {
                    y = 1f - x;
                    i = 0;
                }
                else if (ix >= 0x3e6d3308)
                {
                    y = x - (Tc - 1f);
                    i = 1;
                }
                else
                {
                    y = x;
                    i = 2;
                }
            }
            else
            {
                r = 0;
                if (ix >= 0x3fdda618)
                {
                    y = 2 - x;
                    i = 0;
                }
                else if (ix >= 0x3F9da620)
                {
                    y = x - Tc;
                    i = 1;
                }
                else
                {
                    y = x - 1f;
                    i = 2;
                }
            }
            switch (i)
            {
                case 0:
                    z = y * y;
                    p1 = A0 + z * (A2 + z * A4);
                    p2 = z * (A1 + z * (A3 + z * A5));
                    p = y * p1 + p2;
                    r += p - y / 2;
                    break;
                case 1:
                    p = T0 + y * T1 + y * y * (T2 + y * (T3 + y * (T4 + y * (T5 + y * (T6 + y * T7)))));
                    r += Tf + p;
                    break;
                default:
                    p1 = y * (U0 + y * (U1 + y * U2));
                    p2 = 1f + y * (V1 + y * (V2 + y * V3));
                    r += p1 / p2 - y / 2;
                    break;
            }
        }
        else if (ix < 0x41000000)
        {
            i = (int)x;
            y = x - i;
            p = y * (S0 + y * (S1 + y * (S2 + y * S3)));
            q = 1f + y * (R1 + y * (R2 + y * R3));
            r = y / 2 + p / q;
            z = 1f;
            switch (i)
            {
                case 7: z *= y + 6; goto case 6;
                case 6: z *= y + 5; goto case 5;
                case 5: z *= y + 4; goto case 4;
                case 4: z *= y + 3; goto case 3;
                case 3:
                    z *= y + 2;
                    r += MathF.Log(z);
                    break;
            }
        }
        else if (ix < 0x4d000000)
        {
            var t = MathF.Log(x);
            z = 1f / x;
            y = z * z;
            var w = W0 + z * (W1 + y * W2);
            r = (x - 0.5f) * (t - 1f) + w;
        }
        else
        {
            r = x * (MathF.Log(x) - 1f);
        }
        if (hx < 0) r = nadj - r;
        return r;
    }

    // ---- s_expm1f.c --------------------------------------------------------------------------
    private const float Ef_OThreshold = 8.8721679688e+01f, Ef_Ln2Hi = 6.9313812256e-01f, Ef_Ln2Lo = 9.0580006145e-06f,
        Ef_InvLn2 = 1.4426950216e+00f, Ef_Q1 = -3.3333212137e-2f, Ef_Q2 = 1.5807170421e-3f, Ef_Huge = 1.0e+30f, Ef_Tiny = 1.0e-30f;

    public static float ExpM1F(float x)
    {
        float y, hi, lo, c = 0f, t, e, hxs, hfx, r1, twopk;
        int k;
        var hx = BitConverter.SingleToUInt32Bits(x);
        var xsb = hx & 0x80000000u;
        hx &= 0x7fffffff;

        if (hx >= 0x4195b844)
        {
            if (hx >= 0x42b17218)
            {
                if (hx > 0x7f800000) return x + x;
                if (hx == 0x7f800000) return xsb == 0 ? x : -1.0f;
                if (x > Ef_OThreshold) return float.PositiveInfinity;
            }
            if (xsb != 0)
            {
                if (x + Ef_Tiny < 0.0f) return Ef_Tiny - 1.0f;
            }
        }

        if (hx > 0x3eb17218)
        {
            if (hx < 0x3F851592)
            {
                if (xsb == 0) { hi = x - Ef_Ln2Hi; lo = Ef_Ln2Lo; k = 1; }
                else { hi = x + Ef_Ln2Hi; lo = -Ef_Ln2Lo; k = -1; }
            }
            else
            {
                k = (int)(Ef_InvLn2 * x + (xsb == 0 ? 0.5f : -0.5f));
                t = k;
                hi = x - t * Ef_Ln2Hi;
                lo = t * Ef_Ln2Lo;
            }
            x = hi - lo;
            c = (hi - x) - lo;
        }
        else if (hx < 0x33000000)
        {
            return x;
        }
        else
        {
            k = 0;
        }

        hfx = 0.5f * x;
        hxs = x * hfx;
        r1 = 1.0f + hxs * (Ef_Q1 + hxs * Ef_Q2);
        t = 3.0f - r1 * hfx;
        e = hxs * ((r1 - t) / (6.0f - x * t));
        if (k == 0) return x - (x * e - hxs);
        twopk = BitConverter.UInt32BitsToSingle((uint)(0x7f + k) << 23);
        e = x * (e - c) - c;
        e -= hxs;
        if (k == -1) return 0.5f * (x - e) - 0.5f;
        if (k == 1)
        {
            if (x < -0.25f) return -2.0f * (e - (x + 0.5f));
            return 1.0f + 2.0f * (x - e);
        }
        if (k <= -2 || k > 56)
        {
            y = 1.0f - (e - x);
            if (k == 128) y = y * 2.0f * 1.70141183e+38f;
            else y *= twopk;
            return y - 1.0f;
        }
        if (k < 23)
        {
            t = BitConverter.UInt32BitsToSingle((uint)(0x3f800000 - (0x1000000 >> k)));
            y = t - (e - x);
            y *= twopk;
        }
        else
        {
            t = BitConverter.UInt32BitsToSingle((uint)((0x7f - k) << 23));
            y = x - (e + t);
            y += 1.0f;
            y *= twopk;
        }
        return y;
    }
}
