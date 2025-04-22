from fastapi import FastAPI
from pydantic import BaseModel
from kokoro import KPipeline  # Kokoro TTS 推理库
import numpy as np
import wave
import io
import base64
import os
from contextlib import asynccontextmanager
import logging

import SystemConfig

logger = logging.getLogger(__name__)

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ✅ 启动前执行
    logger.info("FastAPI 启动：is_use_gpu: %s", SystemConfig.is_use_gpu)
    logger.info("FastAPI 启动：is_dev_mode: %s", SystemConfig.is_dev_mode)

    yield  # 🟢 应用运行中

    # ✅ 关闭前执行（可选）
    logger.info("FastAPI 即将关闭")
app = FastAPI(lifespan=lifespan)



# 定义请求数据模型
class TTSRequest(BaseModel):
    text: str
    voice_id: str


@app.post("/synthesize")
async def synthesize_speech(req: TTSRequest):
    # 1. 根据 voice_id 选择对应的 Kokoro 声音模型
    voice_choice = req.voice_id.lower()
    if voice_choice == "male":
        selected_voice = "am_michael"  # 默认选用一个男声音色
    elif voice_choice == "female":
        selected_voice = "af_heart"  # 默认选用一个女声音色
    else:
        selected_voice = voice_choice  # 如果指定了具体声音ID，则直接使用

    # 从 voice_id 前缀确定语言代码，如 'a' 表示美式英语
    lang_code = selected_voice[0]
    # 初始化 Kokoro 推理管线（加载相应语言的模型）
    pipeline = KPipeline(lang_code=lang_code)

    # 2. 利用 Kokoro 模型合成语音
    audio_segments = []
    for _, _, audio in pipeline(req.text, voice=selected_voice):
        # pipeline 返回生成的音频段，可逐段处理
        audio_segments.append(np.array(audio))
    # 将所有音频段拼接为一个完整的音频序列
    audio_data = np.concatenate(audio_segments)

    # 3. 若需要，转换音频采样率为16kHz （模型输出默认为24kHz）
    orig_sr = 24000  # Kokoro 默认输出采样率
    target_sr = 16000
    if orig_sr != target_sr:
        # 简单线性插值进行下采样转换
        old_indices = np.arange(len(audio_data))
        new_length = int(len(audio_data) * target_sr / orig_sr)
        new_indices = np.linspace(0, len(audio_data) - 1, new_length)
        audio_data = np.interp(new_indices, old_indices, audio_data)

    # 4. 将浮点音频数据转换为PCM 16位整数并打包为WAV格式
    audio_int16 = (audio_data * 32767).astype(np.int16)  # 将 -1~1 范围转换为16位整型
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)  # 单声道
        wav_file.setsampwidth(2)  # 16位采样深度 = 2字节
        wav_file.setframerate(target_sr)  # 16kHz 采样率
        wav_file.writeframes(audio_int16.tobytes())
    wav_bytes = buffer.getvalue()

    # 测试使用
    with open("output.wav", "wb") as f:
        f.write(wav_bytes)

    # 5. 将WAV字节数据编码为 base64 字符串，并作为JSON返回
    audio_base64 = base64.b64encode(wav_bytes).decode('utf-8')
    return {"audio": audio_base64}

if __name__ == "__main__":
    import uvicorn

    if SystemConfig.is_dev_mode:
        uvicorn.run(
            "FastVoiceServer:app",
            host="0.0.0.0",
            port=8189,
            reload=SystemConfig.is_dev_mode,
            log_config="log_config.yml"
        )
    else:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8189,
            reload=SystemConfig.is_dev_mode,
            log_config="log_config.yml"
        )