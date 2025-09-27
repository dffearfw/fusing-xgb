import h5py

file_path = "E:/data/CSWE_simp/CSS_SWE_Product_V1.2_Simplified_Version/CSS_SWE_product_V1.2" \
            "/F17_SSMIS_SWE_20130105_DAILY_025KM_V1.2.h5 "

with h5py.File(file_path, 'r') as f:
    print("文件中的数据集:")
    for key in f.keys():
        dataset = f[key]
        print(f"  {key}: shape={dataset.shape}")
        if hasattr(dataset, 'attrs'):
            print(f"    属性: {dict(dataset.attrs)}")
        # 查看前几个值
        if hasattr(dataset, 'shape') and len(dataset.shape) > 0:
            try:
                sample = dataset[:2, :2]  # 前2x2的值
                print(f"    样例值: {sample}")
            except:
                pass
