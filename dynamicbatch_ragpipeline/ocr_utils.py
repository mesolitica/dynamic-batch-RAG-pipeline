from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from enum import auto, Enum
from typing import List, Optional, Tuple, Union
import dataclasses

class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()
    MPT = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "<|im_end|>"
    sep2: str = None
    version: str = "Unknown"

    skip_next: bool = False

    def get_prompt(self):
        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep + '\n'
            for role, message in self.messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        if self.sep_style == SeparatorStyle.MPT:
            if self.system:
                ret = self.system + self.sep 
            else:
                ret = ''
            for role, message in self.messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + message + self.sep
                else:
                    ret += role
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")


    def append_message(self, role, message):
        self.messages.append([role, message])

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2)

class GOTImageEvalProcessor:
    def __init__(self, image_size=384, mean=None, std=None):
        if mean is None:
            mean = (0.48145466, 0.4578275, 0.40821073)
        if std is None:
            std = (0.26862954, 0.26130258, 0.27577711)

        self.normalize = transforms.Normalize(mean, std)

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                self.normalize,
            ]
        )
    def __call__(self, item):
        return self.transform(item)

conv_mpt = Conversation(
    system="""<|im_start|>system
You should follow the instructions carefully and explain your answers in detail.""",
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
    version="mpt",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.MPT,
    sep="<|im_end|>",
)

DEFAULT_IMAGE_TOKEN = '<image>'
DEFAULT_IMAGE_PATCH_TOKEN = '<imgpad>'
DEFAULT_IM_START_TOKEN = '<img>'
DEFAULT_IM_END_TOKEN = '</img>'
image_processor_high =  GOTImageEvalProcessor(image_size=1024)

image_token_len = 256
qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN*image_token_len + DEFAULT_IM_END_TOKEN + '\n'
stop_str = '<|im_end|>'