from pathlib import Path

from pydantic import BaseModel

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


class BaseAgent:
    name: str = "base"
    output_schema: type[BaseModel] | None = None

    def __init__(self, provider):
        self.provider       = provider
        self.prompt_template = self._load_prompt()

    def _load_prompt(self) -> str:
        path = _PROMPTS_DIR / f"{self.name}.txt"
        return path.read_text() if path.exists() else ""

    def _load_named_prompt(self, prompt_name: str) -> str:
        return (_PROMPTS_DIR / f"{prompt_name}.txt").read_text()

    async def _call(
        self,
        prompt: str,
        schema: type[BaseModel],
        system: str = "",
    ) -> BaseModel:
        return await self.provider.complete_json(
            prompt=prompt,
            schema=schema,
            system=system,
        )

    async def run(self, **context) -> BaseModel:
        raise NotImplementedError
