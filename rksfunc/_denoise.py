from vapoursynth import core, VideoNode


def DPIRMDegrain(
    clip: VideoNode,
    dpir_args: dict = {},
    limit_args: dict = {},
    mdg_args: dict = {},
    plane_qtg: bool = True,
    grain_qtg: bool = False,
    prot_dark: bool = False,
    prot_lthr: int = 8000,  # Y value under prot_lthr (in 16 bit) will be considered as dark scene
    prot_hthr: int = 12000,  # A tanh buffer will be created to smooth between prot_lthr and prot_hthr
    check: bool = False,  # return y_dp, y_lm, mdg instead
) -> VideoNode:
    from vapoursynth import YUV, GRAY
    from vsmlrt import DPIR, DPIRModel, Backend
    from mvsfunc import LimitFilter
    from havsfunc import QTGMC, MCTemporalDenoise
    from math import tanh
    from ._resample import yer, mergeuv

    assert clip.format.color_family in [YUV, GRAY]
    origdep = clip.format.bits_per_sample
    if origdep != 16:
        clip = clip.fmtc.bitdepth(bits=16)
    is_yuv = clip.format.color_family == YUV
    y = yer(clip) if is_yuv else clip

    qtg_args = dict(InputType=1, Sharpness=0, SourceMatch=3)
    dpir_preargs = dict(strength=10, backend=Backend.ORT_COREML())
    dpir_preargs.update(dpir_args)
    limit_preargs = dict(thr=3, elast=2)
    limit_preargs.update(limit_args)
    mdg_preargs = dict(refine=True, DCT=6, blksize=32)
    mdg_preargs.update(mdg_args)

    # DPIR & Limit
    y_qt = QTGMC(y, **qtg_args)
    dpi = (y_qt if plane_qtg else y).fmtc.bitdepth(bits=32)
    y_dp = DPIR(clip=dpi, model=DPIRModel.drunet_gray, **dpir_preargs).fmtc.bitdepth(bits=16)
    y_lm = LimitFilter(y, y_dp, **limit_preargs)

    # MDegrain
    dif = core.std.MakeDiff(y, y_dp)
    dif_lm = core.std.MakeDiff(y_lm, y_dp)
    dif_lm = QTGMC(dif_lm, **qtg_args) if grain_qtg else dif_lm
    dif_md = MCTemporalDenoise(dif, p=dif_lm, limit=0, **mdg_preargs)
    mdg_l = core.std.MergeDiff(y_dp, dif_md)

    # protect dark scene
    if prot_dark:
        dif1 = core.std.MakeDiff(y, y_dp)
        dif2 = core.std.MakeDiff(y, mdg_l)
        neutral = 1 << 15
        mdg_d = core.akarin.Expr([dif1, dif2, y_qt, mdg_l], f'x {neutral} - abs y {neutral} - abs < z a ?')
        y_mid = (prot_lthr + prot_hthr) / 2
        y_len = (prot_hthr - prot_lthr) / 4
        lut_y = [(1 + tanh((x - y_mid) / y_len)) / 2 * 65535 for x in range(65536)]
        dmask = core.std.Lut(y_dp, lut=lut_y)
        mdg = core.std.MaskedMerge(mdg_d, mdg_l, dmask)
    else:
        mdg = mdg_l

    mdg = mergeuv(mdg, clip) if is_yuv else mdg
    mdg = mdg.fmtc.bitdepth(bits=origdep)
    if check:
        return y_dp, y_lm, mdg
    else:
        return mdg


def w2xtrt(
    clip: VideoNode,
    noise: int,
    test: bool = False,
    ofmt: bool = False,
    w2xargs: dict = {},
    **args) -> VideoNode:
    from vsmlrt import Waifu2x, Waifu2xModel, Backend
    from ._resample import torgbs, uvsr

    if clip.format.name == "RGBS":
        crgbs = clip
    elif clip.format.name.startswith("YUV444"):
        crgbs = torgbs(clip)
    elif clip.format.name.startswith("YUV420"):
        crgbs = torgbs(uvsr(clip))

    if test:
        w2x = crgbs.w2xnvk.Waifu2x(noise, 1, 2)
    else:
        w2xbe = args.get('w2b', Backend.ORT_COREML())
        preargs = {'backend': w2xbe}
        preargs.update(w2xargs)
        w2x = Waifu2x(clip=crgbs, noise=noise, scale=1, model=Waifu2xModel.anime_style_art, **preargs)
    ofmt = args.get('o420p16', False) or ofmt  # history problem

    if ofmt:
        return w2x.resize.Spline36(format=clip.format.id, matrix=1, dither_type='error_diffusion')
    else:
        return w2x


def TempoStab(clip: VideoNode, mdargs: dict = {}, mdmode=2) -> VideoNode:
    from vapoursynth import YUV, GRAY, Error
    from havsfunc import SMDegrain, QTGMC, MCTemporalDenoise
    from ._resample import yer, mergeuv

    if clip.format.color_family != YUV and clip.format.color_family != GRAY:
        raise Error("rksfunc.tempostab: only YUV and GRAY format are supported.")
    origdep = clip.format.bits_per_sample
    if origdep != 16:
        clip = clip.fmtc.bitdepth(bits=16)
    if clip.format.color_family == YUV:
        clip_y = yer(clip)
    else:
        clip_y = clip
    tref = QTGMC(clip_y, InputType=1, Sharpness=0, SourceMatch=3, opencl=True)
    if mdmode == 1:
        pre_mdargs = {'RefineMotion': True, 'dct': 6, 'blksize': 32, 'prefilter': tref}
        pre_mdargs.update(mdargs)
        smd = SMDegrain(clip_y, **pre_mdargs)
    elif mdmode == 2:
        pre_mdargs = {'p': tref, 'refine': True, 'blksize': 32, 'limit': 0, 'DCT': 6}
        pre_mdargs.update(mdargs)
        smd = MCTemporalDenoise(clip_y, **pre_mdargs)
    else:
        raise ValueError('invalid mdmode value.')
    if clip.format.color_family == YUV:
        smd = mergeuv(smd, clip)
    if origdep != 16:
        smd = smd.fmtc.bitdepth(bits=origdep)
    return smd

# (variable) def Bv2(
#     clip: VideoNode,
#     ref: VideoNode = ...,
#     sigma: float | Sequence[float] = ...,
#     block_step: int | Sequence[int] = ...,
#     bm_range: int | Sequence[int] = ...,
#     radius: int = ...,
#     ps_num: int = ...,
#     ps_range: int = ...,
#     chroma: int = ...,
#     zero_init: int = ...
# ) -> VideoNode
def BM3DRef(
    c420p16: VideoNode, ref: VideoNode,
    bm3d=core.bm3dcpu, chroma=True,
    sigma_Y=2, radius_Y=1, block_step_Y=4, bm_range_Y=8, ps_num_Y=2, ps_range_Y=8,
    sigma_chroma=2, radius_chroma=0, block_step_chroma=4, bm_range_chroma=8, ps_num_chroma=2, ps_range_chroma=6,
) -> VideoNode:
    from vapoursynth import YUV444PS
    from vsutil import split, join
    from ._resample import rgb2opp, opp2rgb, torgbs

    Bv2 = bm3d.BM3Dv2
    half_width = c420p16.width // 2  # half width
    half_height = c420p16.height // 2  # half height
    srcY_float, srcU_float, srcV_float = split(c420p16.fmtc.bitdepth(bits=32))
    refY_float, refU_float, refV_float = split(ref.fmtc.bitdepth(bits=32))
    vfinalY = Bv2(clip=srcY_float, ref=refY_float, sigma=sigma_Y, block_step=block_step_Y, bm_range=bm_range_Y, radius=radius_Y, ps_num=ps_num_Y, ps_range=ps_range_Y)
    vfinalY_half = vfinalY.resize.Spline36(width=half_width, height=half_height, src_left=-0.5)
    refY_half = refY_float.resize.Spline36(width=half_width, height=half_height, src_left=-0.5)
    srchalf_444 = join([vfinalY_half, srcU_float, srcV_float])
    refhalf_444 = join([refY_half, refU_float, refV_float])
    srchalf_opp = rgb2opp(torgbs(srchalf_444))
    refhalf_opp = rgb2opp(torgbs(refhalf_444))
    vfinal_half = Bv2(clip=srchalf_opp, ref=refhalf_opp, sigma=sigma_chroma, block_step=block_step_chroma, bm_range=bm_range_chroma, radius=radius_chroma, ps_num=ps_num_chroma, ps_range=ps_range_chroma, chroma=chroma)
    vfinal_half = opp2rgb(vfinal_half).resize.Spline36(format=YUV444PS, matrix=1)
    _, vfinal_u, vfinal_v = split(vfinal_half)
    vfinal = join([vfinalY, vfinal_u, vfinal_v])
    return vfinal.fmtc.bitdepth(bits=16)

def BM3DWrapper(
    c420p16: VideoNode,
    bm3d=core.bm3dcpu, chroma=True,
    sigma_Y=1.2, radius_Y=1, block_step_Y=4, bm_range_Y=8, ps_num_Y=2, ps_range_Y=8, delta_sigma_Y=0.6,
    sigma_chroma=2.4, radius_chroma=0, block_step_chroma=4, bm_range_chroma=8, ps_num_chroma=2, ps_range_chroma=6, delta_sigma_chroma=1.2,
) -> VideoNode:
    '''
    Note: delta_sigma_xxx is added to sigma_xxx in step basic.
    '''
    from vapoursynth import YUV444PS
    from vsutil import split, join
    from ._resample import rgb2opp, opp2rgb, torgbs

    Bv2 = bm3d.BM3Dv2
    half_width = c420p16.width // 2  # half width
    half_height = c420p16.height // 2  # half height
    srcY_float, srcU_float, srcV_float = split(c420p16.fmtc.bitdepth(bits=32))

    # Explicitly adding parameter names to Bv2 calls
    vbasic_y = Bv2(
        clip=srcY_float,
        ref=srcY_float,
        sigma=sigma_Y + delta_sigma_Y,
        block_step=block_step_Y,
        bm_range=bm_range_Y,
        radius=radius_Y,
        ps_num=ps_num_Y,
        ps_range=ps_range_Y,
    )
    
    vfinal_y = Bv2(
        clip=srcY_float,
        ref=vbasic_y,
        sigma=sigma_Y,
        block_step=block_step_Y,
        bm_range=bm_range_Y,
        radius=radius_Y,
        ps_num=ps_num_Y,
        ps_range=ps_range_Y,
    )
    
    vyhalf = vfinal_y.resize.Spline36(half_width, half_height, src_left=-0.5)
    srchalf_444 = join([vyhalf, srcU_float, srcV_float])
    srchalf_opp = rgb2opp(torgbs(srchalf_444))

    vbasic_half = Bv2(
        clip=srchalf_opp,
        ref=srchalf_opp,
        sigma=sigma_chroma + delta_sigma_chroma,
        block_step=block_step_chroma,
        bm_range=bm_range_chroma,
        radius=radius_chroma,
        ps_num=ps_num_chroma,
        ps_range=ps_range_chroma,
        chroma=chroma,
        zero_init=0
    )

    vfinal_half = Bv2(
        clip=srchalf_opp,
        ref=vbasic_half,
        sigma=sigma_chroma,
        block_step=block_step_chroma,
        bm_range=bm_range_chroma,
        radius=radius_chroma,
        ps_num=ps_num_chroma,
        ps_range=ps_range_chroma,
        chroma=chroma,
        zero_init=0
    )

    vfinal_half = opp2rgb(vfinal_half).resize.Spline36(format=YUV444PS, matrix=1)
    _, vfinal_u, vfinal_v = split(vfinal_half)
    vfinal = join([vfinal_y, vfinal_u, vfinal_v])
    return vfinal.fmtc.bitdepth(bits=16)


def light_vfinal(
    c420p16, s1=1.2, r1=1, bs1=3, br1=12, pn1=2, pr1=8, ds1=0.5,
    s2=2, r2=0, bs2=4, br2=8, pn2=2, pr2=6, ds2=1, bm3d="bm3dcpu", chroma=True
) -> VideoNode:
    return BM3DWrapper(
        c420p16, eval(f"core.{bm3d}"), chroma,
        s1, r1, bs1, br1, pn1, pr1, ds1,
        s2, r2, bs2, br2, pn2, pr2, ds2,
    )


def medium_vfinal(
    c420p16, s1=2.5, r1=1, bs1=3, br1=12, pn1=2, pr1=8,
    s2=2, r2=0, bs2=4, br2=8, pn2=2, pr2=6, ref=None, dftsigma=4, dfttbsize=1,
    bm3d="bm3d", chroma=True
) -> VideoNode:
    if ref is None:
        from havsfunc import EdgeCleaner
        DFTTest = core.dfttest.DFTTest
        ref = DFTTest(c420p16, sigma=dftsigma, tbsize=dfttbsize, planes=[0, 1, 2])
        ref = EdgeCleaner(ref)
    return BM3DRef(
        c420p16, ref, eval(f"core.{bm3d}"), chroma,
        s1, r1, bs1, br1, pn1, pr1,
        s2, r2, bs2, br2, pn2, pr2,
    )


def DeFilmGrain(clip: VideoNode, s1=16, s2=3, s3=3, g=1.5, dark=10000) -> VideoNode:
    from vapoursynth import YUV, GRAY, Error
    from havsfunc import QTGMC
    from vsutil import iterate
    from ._resample import yer, mergeuv
    DFTTest = core.dfttest.DFTTest

    def gamma_curve(clip_y, gamma) -> VideoNode:
        def lut_y(x):
            floa = min(max(int(x / 56283 * 65536), 0), 65535) / 65535
            gammaed = floa ** gamma
            return min(max(int(gammaed * 56283 + 4112), 4112), 60395)

        return core.std.Lut(clip_y, planes=[0], function=lut_y)

    if clip.format.color_family != YUV and clip.format.color_family != GRAY:
        raise Error("rksfunc.defilmgrain: only YUV and GRAY format are supported.")
    origdep = clip.format.bits_per_sample
    if origdep != 16:
        clip = clip.fmtc.bitdepth(bits=16)
    if clip.format.color_family == YUV:
        clip_y = yer(clip)
    else:
        clip_y = clip
    clip_yf = clip_y.fmtc.bitdepth(bits=32)
    basic_f = clip_yf.bm3dcpu.BM3Dv2(clip_yf, s1, 3, 12, 0, 2, 6)
    final_f = clip_yf.bm3dcpu.BM3Dv2(basic_f, s1, 3, 12, 0, 2, 6)
    line_f = clip_yf.bm3dcpu.BM3Dv2(final_f, s2, 3, 15, 0, 2, 8)
    line = line_f.fmtc.bitdepth(bits=16)
    clearplane = DFTTest(final_f.fmtc.bitdepth(bits=16), tbsize=1)
    emask = gamma_curve(clearplane, g).tcanny.TCanny(1, mode=0, op=3)
    emask = iterate(emask, core.std.Maximum, 2)
    emask = iterate(emask, core.std.Inflate, 2)
    emask = core.akarin.Expr([emask, clearplane], f"y {dark} > 0 x ?")
    tref = QTGMC(clip_y, InputType=1, Sharpness=0, SourceMatch=3)
    ttp = clip_y.ttmpsm.TTempSmooth(3, 5, 3, 3, pfclip=tref)
    ttp = core.std.MaskedMerge(ttp, line, emask)
    ttp_f = ttp.fmtc.bitdepth(bits=32)
    vfinal_f = ttp_f.bm3dcpu.BM3Dv2(clip_yf, s3, 3, 12, 1, 2, 8)
    vfinal = vfinal_f.fmtc.bitdepth(bits=16)
    if clip.format.color_family == YUV:
        vfinal = mergeuv(vfinal, clip)
    if origdep != 16:
        vfinal = vfinal.fmtc.bitdepth(bits=origdep)
    return vfinal


def ChromaDenoise(clip: VideoNode, chroma_sr=False, sigma=1.2, bm3d=core.bm3dcpu) -> VideoNode:
    from ._resample import uvsr, half444, mergeuv

    c32 = (uvsr(clip) if chroma_sr else half444(clip)).fmtc.bitdepth(bits=32)
    w2x = w2xtrt(c32, 3, ofmt=True)
    vfn = bm3d.BM3Dv2(c32, w2x, sigma, 3, 8, 1, 2, 8).fmtc.bitdepth(bits=16)
    return mergeuv(clip, vfn)
