from transformers import PretrainedConfig

class SpaceTimeMiniLMConfig(PretrainedConfig):
    model_type = "space-time-minilm"

    def __init__(self, num_space=4, num_time=60, **kwargs):
        super().__init__(**kwargs)
        self.num_space = num_space
        self.num_time = num_time
        self.use_space_embedding = True
        self.use_time_embedding = True