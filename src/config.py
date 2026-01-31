

class CFG:
    DATA_PATH   = "/Users/dric/projects/research/speech/afri-voices/data"
    ASR_MODEL_PATH = "./models/Qwen3-ASR-1.7B"
    FORCED_ALIGNER_PATH = "./models/Qwen3-ForcedAligner-0.6B"

    URL_ZH = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_zh.wav"
    URL_EN = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav"

    DEVICE = "mps" #"cuda:0"
    BATCH_SIZE = 8
    MAX_NEW_TOKENS = 1024
    GPU_MEM_PCT=0.8