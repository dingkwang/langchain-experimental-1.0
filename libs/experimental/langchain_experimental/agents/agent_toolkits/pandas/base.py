"""Agent for working with pandas objects."""

import warnings
from typing import Any, Dict, List, Literal, Optional, Sequence, Union

from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import HumanMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.prompts import ChatPromptTemplate

from langchain_experimental.tools.python.tool import PythonAstREPLTool
from langchain_core.tools import BaseTool
from langchain_core.utils.interactive_env import is_interactive_env

from langchain_experimental.agents.agent_toolkits.pandas.prompt import (
    PREFIX_FUNCTIONS,
    FUNCTIONS_WITH_DF,
    MULTI_DF_PREFIX_FUNCTIONS,
    FUNCTIONS_WITH_MULTI_DF,
)

from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_agent
from langgraph.errors import GraphRecursionError

import pandas as pd


def _get_functions_prompt(
    df: Any,
    *,
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
    include_df_in_prompt: Optional[bool] = True,
    number_of_head_rows: int = 5,
) -> ChatPromptTemplate:
    if isinstance(df, list):
        if include_df_in_prompt:
            dfs_head = "\n\n".join([d.head(number_of_head_rows).to_markdown() for d in df])
            s = (suffix or FUNCTIONS_WITH_MULTI_DF).format(dfs_head=dfs_head)
        else:
            s = suffix or ""
        p = (prefix or MULTI_DF_PREFIX_FUNCTIONS).format(num_dfs=str(len(df)))
    else:
        if include_df_in_prompt:
            df_head = str(df.head(number_of_head_rows).to_markdown())
            s = (suffix or FUNCTIONS_WITH_DF).format(df_head=df_head)
        else:
            s = suffix or ""
        p = prefix or PREFIX_FUNCTIONS

    system_content = p + s

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_content),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    return prompt


class PandasAgentExecutor:
    """Wrapper to mimic legacy AgentExecutor."""

    def __init__(
        self,
        graph,
        verbose: bool = False,
        return_intermediate_steps: bool = False,
        max_iterations: Optional[int] = 15,
        max_execution_time: Optional[float] = None,
        early_stopping_method: str = "force",
        handle_parsing_errors: bool = False,
        **kwargs: Any,
    ):
        self.graph = graph
        self.verbose = verbose
        self.return_intermediate_steps = return_intermediate_steps
        self.max_iterations = max_iterations or 15
        self.max_execution_time = max_execution_time
        self.early_stopping_method = early_stopping_method
        self.handle_parsing_errors = handle_parsing_errors
        if kwargs:
            warnings.warn(f"Ignoring deprecated kwargs: {kwargs}")

    def invoke(
        self, input_dict: Union[str, Dict[str, Any]], config: Optional[Dict] = None, **kwargs: Any
    ) -> Dict[str, Any]:
        if isinstance(input_dict, str):
            input_dict = {"input": input_dict}
        messages = [HumanMessage(content=input_dict.get("input", ""))]
        config = config or {}
        config.setdefault("configurable", {})
        if "configurable" in config and "thread_id" not in config["configurable"]:
            config["configurable"]["thread_id"] = "default_thread"
        if "session_id" in config:
            config["configurable"]["thread_id"] = config.pop("session_id", None)
        recursion_limit = self.max_iterations * 2 + 1
        try:
            result = self.graph.invoke(
                {"messages": messages},
                config=config,
                recursion_limit=recursion_limit,
                **kwargs,
            )
            if self.verbose:
                for msg in result["messages"]:
                    print(msg)
            last_msg = result["messages"][-1]
            output = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
            if self.return_intermediate_steps:
                # Simple approximation: collect tool calls and outputs
                intermediate_steps = []
                msgs = result["messages"]
                i = 0
                while i < len(msgs) - 1:
                    if isinstance(msgs[i], HumanMessage):
                        i += 1
                        continue
                    if hasattr(msgs[i], "tool_calls") and msgs[i].tool_calls:
                        for tc in msgs[i].tool_calls:
                            tool_input = tc["args"]
                            # Next is tool message
                            if i + 1 < len(msgs) and isinstance(msgs[i+1], dict) and "tool_call_id" in msgs[i+1]:
                                observation = msgs[i+1]["content"]
                            else:
                                observation = ""
                            intermediate_steps.append(
                                (
                                    {"tool": tc["name"], "tool_input": tool_input, "log": ""},
                                    observation,
                                )
                            )
                            i += 2
                        continue
                    i += 1
                return {
                    "input": input_dict["input"],
                    "output": output,
                    "intermediate_steps": intermediate_steps,
                }
            return {"input": input_dict["input"], "output": output}
        except GraphRecursionError:
            error_msg = "Agent stopped due to max iterations or recursion limit."
            if self.handle_parsing_errors:
                return {"input": input_dict["input"], "output": error_msg}
            raise
        except Exception as e:
            if self.handle_parsing_errors:
                return {"input": input_dict["input"], "output": f"Error: {str(e)}"}
            raise

    def run(self, input: str, callbacks: Optional[List] = None, **kwargs: Any) -> str:
        """Legacy run method."""
        warnings.warn("run is deprecated, use invoke instead.", DeprecationWarning)
        result = self.invoke({"input": input}, callbacks=callbacks, **kwargs)
        return result["output"]


def create_pandas_dataframe_agent(
    llm: LanguageModelLike,
    df: Any,
    agent_type: Union[Literal["openai-tools", "tool-calling"]] = "tool-calling",
    verbose: bool = False,
    return_intermediate_steps: bool = False,
    max_iterations: Optional[int] = 15,
    max_execution_time: Optional[float] = None,
    early_stopping_method: str = "force",
    include_df_in_prompt: Optional[bool] = True,
    number_of_head_rows: int = 5,
    extra_tools: Sequence[BaseTool] = (),
    engine: Literal["pandas", "modin"] = "pandas",
    allow_dangerous_code: bool = False,
    handle_parsing_errors: bool = False,
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
    agent_executor_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> PandasAgentExecutor:
    """Construct a Pandas agent from an LLM and dataframe(s).

    Security Notice:
        This agent relies on access to a python repl tool which can execute
        arbitrary code. This can be dangerous and requires a specially sandboxed
        environment to be safely used. Failure to run this code in a properly
        sandboxed environment can lead to arbitrary code execution vulnerabilities,
        which can lead to data breaches, data loss, or other security incidents.

        Do not use this code with untrusted inputs, with elevated permissions,
        or without consulting your security team about proper sandboxing!

        You must opt-in to use this functionality by setting allow_dangerous_code=True.

    Args:
        llm: Language model to use for the agent. Expected to support tool calling.
        df: Pandas dataframe or list of Pandas dataframes.
        agent_type: Must be "tool-calling" or "openai-tools".
        verbose: Whether to print intermediate steps.
        return_intermediate_steps: Whether to return intermediate steps.
        max_iterations: Maximum number of iterations.
        max_execution_time: Maximum execution time.
        early_stopping_method: Early stopping method.
        include_df_in_prompt: Whether to include dataframe head in prompt.
        number_of_head_rows: Number of head rows to include.
        extra_tools: Additional tools.
        engine: "pandas" or "modin".
        allow_dangerous_code: Opt-in for dangerous code execution.
        handle_parsing_errors: Handle parsing errors gracefully.
        prefix: Custom prefix for prompt.
        suffix: Custom suffix for prompt.
        agent_executor_kwargs: Additional kwargs for executor (ignored).
        **kwargs: Deprecated.

    Returns:
        A PandasAgentExecutor compatible with legacy AgentExecutor interface.
    """
    if agent_type not in ("tool-calling", "openai-tools"):
        raise ValueError(
            "Only 'tool-calling' and 'openai-tools' are supported in LangChain 1.0. "
            "Legacy types like 'zero-shot-react-description' are not supported. "
            "See migration guide for details."
        )

    if not allow_dangerous_code:
        raise ValueError(
            "This agent relies on access to a python repl tool which can execute "
            "arbitrary code. This can be dangerous and requires a specially sandboxed "
            "environment to be safely used. Please read the security notice in the "
            "doc-string of this function. You must opt-in to use this functionality "
            "by setting allow_dangerous_code=True."
            "For general security guidelines, please see: "
            "https://python.langchain.com/docs/security/"
        )

    try:
        if engine == "modin":
            import modin.pandas as pd
        elif engine == "pandas":
            import pandas as pd
        else:
            raise ValueError(
                f"Unsupported engine {engine}. It must be one of 'modin' or 'pandas'."
            )
    except ImportError as e:
        raise ImportError(
            f"`{engine}` package not found, please install with `pip install {engine}`"
        ) from e

    if is_interactive_env():
        pd.set_option("display.max_columns", None)

    for _df in df if isinstance(df, list) else [df]:
        if not isinstance(_df, pd.DataFrame):
            raise ValueError(f"Expected pandas DataFrame, got {type(_df)}")

    if prefix is not None and suffix is not None:
        warnings.warn("Both prefix and suffix provided; suffix will override defaults.")

    df_locals = {}
    if isinstance(df, list):
        for i, dataframe in enumerate(df):
            df_locals[f"df{i + 1}"] = dataframe
    else:
        df_locals["df"] = df

    tools = [PythonAstREPLTool(locals=df_locals)] + list(extra_tools)

    prompt = _get_functions_prompt(
        df,
        prefix=prefix,
        suffix=suffix,
        include_df_in_prompt=include_df_in_prompt,
        number_of_head_rows=number_of_head_rows,
    )

    def state_modifier(state: dict) -> list:
        from langchain_core.messages import BaseMessage
        system_messages = prompt.invoke({"messages": []}).to_messages()
        return system_messages + state["messages"]

    checkpointer = MemorySaver()
    graph = create_agent(
        llm,
        tools,
        prompt=state_modifier,
        checkpointer=checkpointer,
    )

    executor_kwargs = agent_executor_kwargs or {}
    executor = PandasAgentExecutor(
        graph,
        verbose=verbose,
        return_intermediate_steps=return_intermediate_steps,
        max_iterations=max_iterations,
        max_execution_time=max_execution_time,
        early_stopping_method=early_stopping_method,
        handle_parsing_errors=handle_parsing_errors,
        **executor_kwargs,
    )

    if kwargs:
        warnings.warn(
            f"Received additional deprecated kwargs {list(kwargs.keys())} which are ignored."
        )

    return executor