"""
Prompt formatting templates for LLM fine-tuning.
"""
from typing import Dict, Any, Optional


class PromptTemplate:
    """Base class for prompt templates."""
    
    def format(self, **kwargs) -> str:
        """Format the prompt with given arguments."""
        raise NotImplementedError


class AlpacaTemplate(PromptTemplate):
    """Alpaca instruction template."""
    
    def format(self, instruction: str, input: str = "", output: str = "") -> str:
        """
        Format in Alpaca style.
        
        Args:
            instruction: The instruction
            input: Optional input context
            output: Optional output (for training)
        
        Returns:
            Formatted prompt
        """
        if input:
            prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""
        else:
            prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{output}"""
        
        return prompt


class ChatMLTemplate(PromptTemplate):
    """ChatML template (used by many chat models)."""
    
    def format(self, system: str = "", user: str = "", assistant: str = "") -> str:
        """
        Format in ChatML style.
        
        Args:
            system: System message
            user: User message
            assistant: Assistant response
        
        Returns:
            Formatted prompt
        """
        messages = []
        
        if system:
            messages.append(f"<|im_start|>system\n{system}<|im_end|>")
        
        if user:
            messages.append(f"<|im_start|>user\n{user}<|im_end|>")
        
        if assistant:
            messages.append(f"<|im_start|>assistant\n{assistant}<|im_end|>")
        
        return "\n".join(messages)


class Llama2ChatTemplate(PromptTemplate):
    """Llama-2 chat template."""
    
    def format(self, system: str = "", user: str = "", assistant: str = "") -> str:
        """
        Format in Llama-2 chat style.
        
        Args:
            system: System message
            user: User message
            assistant: Assistant response
        
        Returns:
            Formatted prompt
        """
        if system:
            prompt = f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user} [/INST] {assistant}"
        else:
            prompt = f"<s>[INST] {user} [/INST] {assistant}"
        
        return prompt


class SimpleCompletionTemplate(PromptTemplate):
    """Simple text completion template."""
    
    def format(self, text: str = "", completion: str = "") -> str:
        """
        Simple concatenation for completion tasks.
        
        Args:
            text: Input text/prompt
            completion: Completion text
        
        Returns:
            Formatted prompt
        """
        if completion:
            return f"{text}{completion}"
        return text


# Template registry
TEMPLATES = {
    "alpaca": AlpacaTemplate(),
    "chatml": ChatMLTemplate(),
    "llama2": Llama2ChatTemplate(),
    "completion": SimpleCompletionTemplate()
}


def get_template(template_name: str) -> PromptTemplate:
    """
    Get a prompt template by name.
    
    Args:
        template_name: Template name
    
    Returns:
        PromptTemplate instance
    """
    if template_name not in TEMPLATES:
        raise ValueError(f"Unknown template: {template_name}. Available: {list(TEMPLATES.keys())}")
    
    return TEMPLATES[template_name]


def format_prompt(template_name: str, **kwargs) -> str:
    """
    Format a prompt using a template.
    
    Args:
        template_name: Template name
        **kwargs: Template-specific arguments
    
    Returns:
        Formatted prompt
    """
    template = get_template(template_name)
    return template.format(**kwargs)


def apply_template_to_dataset(
    data: list[Dict[str, Any]],
    template_name: str,
    field_mapping: Dict[str, str]
) -> list[str]:
    """
    Apply a template to a dataset.
    
    Args:
        data: List of data dictionaries
        template_name: Template name
        field_mapping: Mapping from template fields to data fields
            e.g., {"instruction": "question", "output": "answer"}
    
    Returns:
        List of formatted prompts
    """
    template = get_template(template_name)
    formatted = []
    
    for item in data:
        # Map fields
        template_args = {
            template_field: item.get(data_field, "")
            for template_field, data_field in field_mapping.items()
        }
        
        # Format
        formatted.append(template.format(**template_args))
    
    return formatted
