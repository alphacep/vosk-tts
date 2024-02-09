from transformers import HubertModel

class HubertModelWithFinalProj(HubertModel):
    def __init__(self, config):
        super().__init__(config)

        # The final projection layer is only used for backward compatibility.
        # Following https://github.com/auspicious3000/contentvec/issues/6
        # Remove this layer is necessary to achieve the desired outcome.
        self.final_proj = torch.nn.Linear(config.hidden_size, config.classifier_proj_size)

    def extract(self, x):
        return self(x)["last_hidden_state"]
