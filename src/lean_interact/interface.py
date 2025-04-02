from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

# Classes and attributes are aligned with the Lean REPL: https://github.com/leanprover-community/repl/blob/2f0a3cb876b045cc0fe550ca3a625bc479816739/REPL/JSON.lean


class REPLBaseModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow", populate_by_name=True)

    def __repr__(self) -> str:
        """Return string representation showing only set attributes."""
        attrs = []
        for name in self.__pydantic_fields_set__:
            attrs.append(f"{name}={getattr(self, name)!r}")
        return f"{self.__class__.__name__}({', '.join(attrs)})"

    def __str__(self) -> str:
        """Return simplified string showing only set attributes."""
        return self.__repr__()


# Request


class BaseREPLQuery(REPLBaseModel):
    """Base class for all Lean requests."""


class CommandOptions(REPLBaseModel):
    all_tactics: Annotated[bool | None, Field(alias="allTactics")] = None
    infotree: str | None = None


class Command(BaseREPLQuery, CommandOptions):
    cmd: Annotated[str, Field(min_length=1)]
    env: int | None = None


class FileCommand(BaseREPLQuery, CommandOptions):
    path: Annotated[str, Field(min_length=1)]


class ProofStep(BaseREPLQuery, REPLBaseModel):
    proof_state: Annotated[int, Field(alias="proofState")]
    tactic: Annotated[str, Field(min_length=1)]


class PickleEnvironment(BaseREPLQuery):
    env: int
    pickle_to: Annotated[str, Field(min_length=1, alias="pickleTo")]


class UnpickleEnvironment(BaseREPLQuery):
    unpickle_env_from: Annotated[str, Field(min_length=1, alias="unpickleEnvFrom")]


class PickleProofState(BaseREPLQuery):
    proof_state: Annotated[int, Field(alias="proofState")]
    pickle_to: Annotated[str, Field(min_length=1, alias="pickleTo")]


class UnpickleProofState(BaseREPLQuery):
    unpickle_proof_state_from: Annotated[str, Field(min_length=1, alias="unpickleProofStateFrom")]
    env: int | None = None


# Intermediate classes


class Pos(REPLBaseModel):
    line: int
    column: int

    def __le__(self, other: "Pos") -> bool:
        if self.line < other.line:
            return True
        if self.line == other.line:
            return self.column <= other.column
        return False

    def __lt__(self, other: "Pos") -> bool:
        return self <= other and not self == other


class Message(REPLBaseModel):
    start_pos: Annotated[Pos, Field(alias="pos")]
    end_pos: Annotated[Pos | None, Field(alias="endPos")] = None
    severity: Literal["error", "warning", "info", "trace"]
    data: str


class Sorry(REPLBaseModel):
    start_pos: Annotated[Pos, Field(alias="pos")]
    end_pos: Annotated[Pos, Field(alias="endPos")]
    goal: str
    proof_state: Annotated[int | None, Field(alias="proofState")] = None


class Tactic(REPLBaseModel):
    start_pos: Annotated[Pos, Field(alias="pos")]
    end_pos: Annotated[Pos, Field(alias="endPos")]
    goals: str
    tactic: str
    proof_state: Annotated[int | None, Field(alias="proofState")] = None
    used_constants: Annotated[list[str], Field(default_factory=list, alias="usedConstants")]


def message_intersects_code(msg: Message | Sorry, start_pos: Pos | None, end_pos: Pos | None) -> bool:
    res = True
    if start_pos is not None and msg.end_pos is not None:
        res = res and msg.end_pos.line >= start_pos.line
    if end_pos is not None and msg.start_pos is not None:
        res = res and msg.start_pos.line <= end_pos.line
    return res


# Response


class BaseREPLResponse(REPLBaseModel):
    messages: list[Message] = Field(default_factory=list)
    sorries: list[Sorry] = Field(default_factory=list)

    def __init__(self, **data):
        if self.__class__ == BaseREPLResponse:
            raise TypeError("BaseResponse cannot be instantiated directly")
        super().__init__(**data)

    def get_errors(self) -> list[Message]:
        """Return all error messages"""
        return [msg for msg in self.messages if msg.severity == "error"]

    def get_warnings(self) -> list[Message]:
        """Return all warning messages"""
        return [msg for msg in self.messages if msg.severity == "warning"]

    def has_errors(self) -> bool:
        """Check if response contains any error messages"""
        return any(msg.severity == "error" for msg in self.messages)

    def lean_code_is_valid(
        self,
        start_pos: Pos | None = None,
        end_pos: Pos | None = None,
        allow_sorry: bool = True,
    ) -> bool:
        """Check if the submitted code is valid Lean code."""
        # check only the messages intersecting the code
        errors = [
            message
            for message in self.messages
            if message_intersects_code(message, start_pos, end_pos) and message.severity == "error"
        ]
        sorries = [message for message in self.sorries if message_intersects_code(message, start_pos, end_pos)]
        return not errors and (allow_sorry or not sorries)


class CommandResponse(BaseREPLResponse):
    env: int
    tactics: list[Tactic] = Field(default_factory=list)
    infotree: dict | None = None


class ProofStepResponse(BaseREPLResponse):
    proof_state: Annotated[int, Field(alias="proofState")]
    goals: list[str] = Field(default_factory=list)
    traces: list[str] = Field(default_factory=list)


class LeanError(REPLBaseModel):
    message: str = ""
