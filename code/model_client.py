import anthropic
from config import settings
from schema import ClaimPrediction


client = anthropic.Anthropic(api_key=settings.anthropic_api_key)


def predict(system_prompt: str, user_content: list[dict]) -> tuple[ClaimPrediction, dict]:
    response = client.messages.parse(
        model=settings.llm_model_name,
        max_tokens=settings.max_tokens,
        system=[{"type": "text", "text": system_prompt,
                   "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user_content}],
        output_format=ClaimPrediction,
    )
    usage = {
          "input": response.usage.input_tokens,
          "output": response.usage.output_tokens,
          "cache_read": response.usage.cache_read_input_tokens,
          "cache_write": response.usage.cache_creation_input_tokens,
      }
    return response.parsed_output, usage