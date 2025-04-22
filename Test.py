from kokoro import KPipeline
from IPython.display import display, Audio
import soundfile as sf
import torch

REPO_ID = 'hexgrad/Kokoro-82M-v1.1-zh'

pipeline = KPipeline(lang_code='z', repo_id=REPO_ID,)
text = '''
刘一辰很帅，很强，超级厉害，是个绝对的天才。
'''
generator = pipeline(text, voice='zf_xiaoxiao')
for i, (gs, ps, audio) in enumerate(generator):
    print(i, gs, ps)
    display(Audio(data=audio, rate=24000, autoplay=i==0))
    sf.write(f'{i}.wav', audio, 24000)