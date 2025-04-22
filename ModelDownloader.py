# download_models.py
from kokoro import download
from kokoro.utils import get_model_dir
import shutil
from pathlib import Path

# 你要保存到的目标目录
custom_dir = Path("./kokoro_models")
custom_dir.mkdir(parents=True, exist_ok=True)

# 你需要的模型列表（也可以添加更多）
model_list = ["zh", "zf_xiaoxiao", "zm_yunyang"]

# 下载模型并复制到目标目录
for model in model_list:
    download(model)  # 下载到默认路径
    src_path = Path(get_model_dir()) / model
    dest_path = custom_dir / model
    if not dest_path.exists():
        shutil.copytree(src_path, dest_path)
        print(f"✅ 模型 {model} 已复制到 {dest_path}")
    else:
        print(f"✅ 模型 {model} 已存在于 {dest_path}")
