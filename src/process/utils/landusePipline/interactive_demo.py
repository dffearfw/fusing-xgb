# interactive_demo.py
from src.process.utils.landusePipline.core.pipeline_builder import LandUsePipelineBuilder

# 方式1: 流畅接口
pipeline = (LandUsePipelineBuilder()
           .fix_data("v1")
           .encode_features("hashing", n_features=8)
           .encode_features("frequency")  # 可以添加多个编码步骤
           .build())

result = pipeline.execute("data.xlsx")

# 方式2: 预定义配方
from src.process.utils.landusePipline.core.pipeline_builder import PipelineRecipes
pipeline = PipelineRecipes.advanced_analysis()
result = pipeline.execute("data.xlsx")

# 方式3: 逐步构建
builder = LandUsePipelineBuilder()
builder.fix_data("v2", aggressive_clean=True)
builder.encode_features("composite", methods=['hashing', 'target'])
pipeline = builder.build()
result = pipeline.execute("data.xlsx")