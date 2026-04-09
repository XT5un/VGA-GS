try:
    from diff_gaussian_rasterization_depth import GaussianRasterizationSettings, GaussianRasterizer
except:
    raise ModuleNotFoundError("not found diff_gaussian_rasterization_depth")

try:
    from diff_gaussian_rasterization_origin import GaussianRasterizationSettings, GaussianRasterizer
except:
    raise ModuleNotFoundError("not found diff_gaussian_rasterization_origin")

try:
    from diff_gaussian_rasterization_2dgs import GaussianRasterizationSettings, GaussianRasterizer
except:
    raise ModuleNotFoundError("not found diff_gaussian_rasterization_2dgs")

try:
    from diff_gaussian_rasterization_RaDe import GaussianRasterizationSettings, GaussianRasterizer
except:
    raise ModuleNotFoundError("not found diff_gaussian_rasterization_RaDe")

print("All modules are imported successfully!")
