"""
文本乱码检测 FastAPI 服务
提供HTTP接口进行文本乱码检测
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict
import time
from datetime import datetime

# 导入检测器
from garbled_text_detector import detect_gibberish, preload_models, CONFIG

# 创建FastAPI应用
app = FastAPI(
    title="文本乱码检测API", description="检测文本中是否包含乱码", version="1.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 请求模型
class TextDetectionRequest(BaseModel):
    """文本检测请求"""

    text: str = Field(..., min_length=1, max_length=50000, description="需要检测的文本")
    detailed: bool = Field(default=True, description="是否返回详细指标")
    llm_confirm: bool = Field(
        default=False, description="是否启用LLM二次确认（当判断为乱码时）"
    )
    llm_token: Optional[str] = Field(
        default=None, description="LLM API Token（启用llm_confirm时需要）"
    )

    class Config:
        schema_extra = {
            "example": {
                "text": "这是一段需要检测的文本",
                "detailed": True,
                "llm_confirm": False,
            }
        }


# 响应模型
class DetectionResult(BaseModel):
    """检测结果"""

    is_gibberish: bool = Field(..., description="是否为乱码")
    confidence: float = Field(..., ge=0, le=1, description="置信度")
    overall_score: float = Field(..., ge=0, le=1, description="综合得分")
    language: str = Field(..., description="检测到的语言")
    processing_time_ms: Optional[float] = Field(None, description="处理时间(毫秒)")
    metrics: Optional[Dict] = Field(None, description="详细指标")
    issues: Optional[list] = Field(None, description="检测到的问题")
    llm_confirmed: Optional[bool] = Field(
        None, description="LLM二次确认结果（仅当启用llm_confirm时）"
    )


# API端点


@app.on_event("startup")
async def startup_event():
    """启动时预加载模型"""
    try:
        preload_models()
        print("[OK] API服务启动成功")
    except Exception as e:
        print(f"[ERROR] 模型加载失败: {e}")


@app.get("/")
async def root():
    """API信息"""
    return {
        "service": "文本乱码检测API",
        "version": "1.0.0",
        "status": "运行中",
        "docs": "http://localhost:8000/docs",
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/config")
async def get_config():
    """获取配置信息"""
    return {"weights": CONFIG["WEIGHTS"], "thresholds": CONFIG["THRESHOLDS"]}


@app.post("/detect", response_model=DetectionResult)
async def detect_text(request: TextDetectionRequest):
    """
    检测文本是否包含乱码

    - **text**: 需要检测的文本
    - **detailed**: 是否返回详细指标
    - **llm_confirm**: 是否启用LLM二次确认（当判断为乱码时调用LLM再次确认）
    - **llm_token**: LLM API Token（启用llm_confirm时需要）
    """
    start_time = time.time()

    try:
        # 执行检测
        result = detect_gibberish(
            request.text, request.detailed, request.llm_confirm, request.llm_token
        )

        # 添加处理时间
        processing_time = (time.time() - start_time) * 1000
        result["processing_time_ms"] = round(processing_time, 2)

        return DetectionResult(**result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"检测出错: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    print("=" * 50)
    print("文本乱码检测API服务")
    print("=" * 50)
    print("访问 http://localhost:8000/docs 查看API文档")
    print("访问 http://localhost:8000 查看服务信息")
    print("=" * 50)

    # 运行服务
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False, log_level="info")
