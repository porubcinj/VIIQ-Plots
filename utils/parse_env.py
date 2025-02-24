import dotenv

class Config:
    def __init__(self):
        self._cuda: bool = True
        self._model: str
        self._polar_height: int = 64
        self._polar_width: int = 512
        self._csv: str
        self._skip_templates: bool = False

    @property
    def CUDA(self): return self._cuda
    @CUDA.setter
    def CUDA(self, value): self._cuda = str(value).lower() == "true"

    @property
    def MODEL(self): return self._model
    @MODEL.setter
    def MODEL(self, value): self._model = str(value)

    @property
    def POLAR_HEIGHT(self): return self._polar_height
    @POLAR_HEIGHT.setter
    def POLAR_HEIGHT(self, value): self._polar_height = int(value)

    @property
    def POLAR_WIDTH(self): return self._polar_width
    @POLAR_WIDTH.setter
    def POLAR_WIDTH(self, value): self._polar_width = int(value)

    @property
    def CSV(self): return self._csv
    @CSV.setter
    def CSV(self, value): self._csv = str(value)

    @property
    def SKIP_TEMPLATES(self): return self._skip_templates
    @SKIP_TEMPLATES.setter
    def SKIP_TEMPLATES(self, value): self._skip_templates = str(value).lower() == "true"

def parse_env(env: str = ".env") -> Config:
    config = Config()
    for key, value in dotenv.dotenv_values(env).items():
        setattr(config, key, value)
    return config