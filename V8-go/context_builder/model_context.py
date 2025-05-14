from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import json

class TokenCounter:
    """Utility class for token counting and management"""
    
    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Rough estimation of tokens (approximately 4 chars per token)"""
        return len(text) // 4

    @staticmethod
    def validate_token_limit(text: str, limit: int) -> bool:
        """Check if text exceeds token limit"""
        return TokenCounter.estimate_tokens(text) <= limit

class TokenLimits:
    """Token limits for different context categories for Gemini"""
    SYSTEM_CONTEXT = 1000
    CODEBASE_METADATA = 2000
    GLOBAL_CONTEXT = 3000
    LOCAL_CONTEXT = 4000
    CHANGE_REQUEST = 2000
    TOTAL_CONTEXT = 32000  # Gemini Pro's context window

@dataclass
class SystemContext:
    """Initial system instructions and context"""
    system_prompt: str
    model_capabilities: List[str]
    behavior_guidelines: List[str]
    response_format: Dict[str, str]
    token_count: int = 0

    def __post_init__(self):
        self.update_token_count()

    def update_token_count(self):
        """Calculate and update token count"""
        content = json.dumps({
            "system_prompt": self.system_prompt,
            "model_capabilities": self.model_capabilities,
            "behavior_guidelines": self.behavior_guidelines,
            "response_format": self.response_format
        })
        self.token_count = TokenCounter.estimate_tokens(content)
        if self.token_count > TokenLimits.SYSTEM_CONTEXT:
            raise ValueError(f"System context exceeds token limit of {TokenLimits.SYSTEM_CONTEXT}")

@dataclass
class CodebaseMetadata:
    """File and codebase metadata with design overview"""
    repository_name: str
    primary_language: str
    file_path: str
    file_type: str
    last_modified: datetime
    dependencies: List[str]
    # New design-related fields
    architecture_pattern: str  # e.g., "MVC", "Microservices", "Layered"
    design_overview: Dict[str, str]  # High-level design descriptions
    key_components: List[Dict[str, str]]  # Major system components
    tech_stack: Dict[str, List[str]]  # Technology stack by category
    token_count: int = 0

    def __post_init__(self):
        self.update_token_count()

    def update_token_count(self):
        content = json.dumps(self.__dict__)
        self.token_count = TokenCounter.estimate_tokens(content)
        if self.token_count > TokenLimits.CODEBASE_METADATA:
            raise ValueError(f"Codebase metadata exceeds token limit of {TokenLimits.CODEBASE_METADATA}")

@dataclass
class GlobalContext:
    """Global context and abstractions"""
    project_architecture: Dict[str, str]
    design_patterns: List[str]
    shared_components: List[str]
    global_variables: Dict[str, str]
    token_count: int = 0

    def __post_init__(self):
        self.update_token_count()

    def update_token_count(self):
        content = json.dumps(self.__dict__)
        self.token_count = TokenCounter.estimate_tokens(content)
        if self.token_count > TokenLimits.GLOBAL_CONTEXT:
            raise ValueError(f"Global context exceeds token limit of {TokenLimits.GLOBAL_CONTEXT}")

@dataclass
class LocalContext:
    """Local context specific to the current task"""
    current_file_content: str
    surrounding_code: Dict[str, str]
    variable_scope: Dict[str, str]
    function_definitions: List[str]
    token_count: int = 0

    def __post_init__(self):
        self.update_token_count()

    def update_token_count(self):
        content = json.dumps(self.__dict__)
        self.token_count = TokenCounter.estimate_tokens(content)
        if self.token_count > TokenLimits.LOCAL_CONTEXT:
            raise ValueError(f"Local context exceeds token limit of {TokenLimits.LOCAL_CONTEXT}")

@dataclass
class ChangeRequest:
    """Code changes or feature requests"""
    request_type: str
    description: str
    affected_files: List[str]
    proposed_changes: Dict[str, str]
    acceptance_criteria: List[str]
    token_count: int = 0

    def __post_init__(self):
        self.update_token_count()

    def update_token_count(self):
        content = json.dumps(self.__dict__)
        self.token_count = TokenCounter.estimate_tokens(content)
        if self.token_count > TokenLimits.CHANGE_REQUEST:
            raise ValueError(f"Change request exceeds token limit of {TokenLimits.CHANGE_REQUEST}")

class ModelContext:
    """Complete context window manager for Gemini"""
    
    def __init__(
        self,
        system: SystemContext,
        codebase: CodebaseMetadata,
        global_context: GlobalContext,
        local_context: LocalContext,
        change_request: ChangeRequest
    ):
        self.system = system
        self.codebase = codebase
        self.global_context = global_context
        self.local_context = local_context
        self.change_request = change_request
        self.validate_total_tokens()

    def get_total_tokens(self) -> int:
        """Calculate total tokens across all context categories"""
        return sum([
            self.system.token_count,
            self.codebase.token_count,
            self.global_context.token_count,
            self.local_context.token_count,
            self.change_request.token_count
        ])

    def validate_total_tokens(self):
        """Ensure total tokens don't exceed Gemini's context window"""
        total_tokens = self.get_total_tokens()
        if total_tokens > TokenLimits.TOTAL_CONTEXT:
            raise ValueError(
                f"Total context ({total_tokens} tokens) exceeds "
                f"Gemini's limit of {TokenLimits.TOTAL_CONTEXT} tokens"
            )

    def get_context_distribution(self) -> Dict[str, float]:
        """Return percentage distribution of tokens across categories"""
        total = self.get_total_tokens()
        return {
            "system": (self.system.token_count / total) * 100,
            "codebase": (self.codebase.token_count / total) * 100,
            "global": (self.global_context.token_count / total) * 100,
            "local": (self.local_context.token_count / total) * 100,
            "change": (self.change_request.token_count / total) * 100
        }

    def to_prompt(self) -> str:
        """Convert context to a formatted prompt string"""
        prompt = {
            "system": {
                "prompt": self.system.system_prompt,
                "capabilities": self.system.model_capabilities,
                "guidelines": self.system.behavior_guidelines
            },
            "codebase": {
                "repository": self.codebase.repository_name,
                "language": self.codebase.primary_language,
                "architecture": self.codebase.architecture_pattern,
                "design": self.codebase.design_overview,
                "components": self.codebase.key_components
            },
            "global_context": {
                "architecture": self.global_context.project_architecture,
                "patterns": self.global_context.design_patterns,
                "shared": self.global_context.shared_components
            },
            "local_context": {
                "current_file": self.local_context.current_file_content,
                "scope": self.local_context.variable_scope,
                "functions": self.local_context.function_definitions
            },
            "change_request": {
                "type": self.change_request.request_type,
                "description": self.change_request.description,
                "affected_files": self.change_request.affected_files
            }
        }
        
        return json.dumps(prompt, indent=2)

    def update_local_context(self, file_content: str, scope: Dict[str, str], functions: List[str]) -> None:
        """Update local context with new file information"""
        self.local_context = LocalContext(
            current_file_content=file_content,
            surrounding_code={},
            variable_scope=scope,
            function_definitions=functions
        )
        self.validate_total_tokens()
