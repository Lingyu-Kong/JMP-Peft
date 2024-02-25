import contextlib
import warnings
from abc import ABC
from logging import getLogger
from types import TracebackType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Generic,
    Mapping,
    MutableMapping,
    TypedDict,
    cast,
)

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import (
    ParamSpec,
    Self,
    TypeVar,
    dataclass_transform,
    deprecated,
    override,
)

log = getLogger(__name__)


class _MISSING:
    pass


MISSING = cast(Any, _MISSING())

TConfig = TypeVar("TConfig", bound="TypedConfig")
P = ParamSpec("P")


def _deep_validate(config: TConfig, strict: bool = True) -> TConfig:
    # First, we call __pre_init__ to allow the config to modify itself a final time
    config.__pre_init__()

    # Then, we dump the config to a dict and then re-validate it
    config_dict = config.pydantic_model().model_dump(round_trip=True)
    # We need to remove the _typed_config_builder_context from the config_dict
    #   because we're no longer a builder config
    _ = config_dict.pop("_typed_config_builder_context", None)

    return cast(
        TConfig,
        config.pydantic_model().model_validate(config_dict, strict=strict),
    )


class _ConfigBuilderContext(TypedDict):
    strict: bool


class ConfigBuilder(contextlib.AbstractContextManager, Generic[TConfig]):
    def __init__(
        self,
        config_cls: Callable[P, TConfig],
        /,
        strict: bool = True,
        *_args: P.args,
        **kwargs: P.kwargs,
    ):
        assert isinstance(config_cls, type), "config_cls must be a class"
        assert not len(
            _args
        ), f"Only keyword arguments are supported for config classes. Got {_args=}."

        self.__config_cls = cast(type[TConfig], config_cls)
        self.__strict = strict
        self.__model_kwargs = kwargs
        self.__exit_stack = contextlib.ExitStack()
        self.__warning_list: list[warnings.WarningMessage] | None = None

    def build(self, config: TConfig) -> TConfig:
        return _deep_validate(config, strict=self.__strict)

    __call__ = build

    @override
    def __enter__(self) -> tuple[Self, TConfig]:
        ctx = _ConfigBuilderContext(strict=self.__strict)

        self.__warning_list = self.__exit_stack.enter_context(
            warnings.catch_warnings(record=True)
        )
        config = cast(
            TConfig,
            self.__config_cls.pydantic_model_cls().model_construct(
                _typed_config_builder_context=ctx,
                **self.__model_kwargs,
            ),
        )
        return self, config

    @override
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ):
        return_value = self.__exit_stack.__exit__(exc_type, exc_value, traceback)
        if warning_list := self.__warning_list:
            for warning in warning_list:
                if (
                    isinstance(warning.message, UserWarning)
                    and "pydantic" in warning.message.args[0].lower()
                ):
                    continue

                warnings.showwarning(
                    message=warning.message,
                    category=warning.category,
                    filename=warning.filename,
                    lineno=warning.lineno,
                    file=warning.file,
                    line=warning.line,
                )

        return return_value


_BaseModelBase = BaseModel
if TYPE_CHECKING:
    _BaseModelBase = ABC

_MutableMappingBase = MutableMapping[str, Any]
if TYPE_CHECKING:
    _MutableMappingBase = object


@dataclass_transform(
    kw_only_default=True,
    field_specifiers=(Field,),
)
class TypedConfig(_BaseModelBase, _MutableMappingBase):
    MISSING: ClassVar[Any] = MISSING
    model_config: ClassVar[ConfigDict] = ConfigDict(
        # By default, Pydantic will throw a warning if a field starts with "model_",
        # so we need to disable that warning (beacuse "model_" is a popular prefix for ML).
        protected_namespaces=(),
        validate_assignment=True,
        strict=True,
        revalidate_instances="always",
        arbitrary_types_allowed=True,
    )

    @classmethod
    def pydantic_model_cls(cls) -> type[BaseModel]:
        return cast(type[BaseModel], cls)

    def pydantic_model(self) -> BaseModel:
        return cast(BaseModel, self)

    @classmethod
    def from_dict(cls, model_dict: Mapping[str, Any]):
        return cast(Self, cls.pydantic_model_cls().model_validate(model_dict))

    @classmethod
    def model_deep_validate(cls, config: TConfig, strict: bool = True) -> TConfig:
        return _deep_validate(config, strict=strict)

    @classmethod
    def builder(cls: type[TConfig], /, strict: bool = True):
        return ConfigBuilder[TConfig](cls, strict=strict)

    def __pre_init__(self):
        """Called before the final config is validated, but after the config is created and populated."""
        pass

    def __post_init__(self):
        """Called after the final config is validated."""
        pass

    if not TYPE_CHECKING:
        _typed_config_builder_context: _ConfigBuilderContext | None = None

        @override
        def model_post_init(  # pyright: ignore[reportGeneralTypeIssues]
            self, __context: Any
        ) -> None:
            super().model_post_init(  # pyright: ignore[reportAttributeAccessIssue]
                __context
            )

            # This fixes the issue w/ `copy.deepcopy` not working properly when
            # the object was created using `cls.model_construct`.
            if not hasattr(self, "__pydantic_private__"):
                object.__setattr__(self, "__pydantic_private__", None)

            # If we're not in a builder, call __post_init__
            if self._typed_config_builder_context is None:
                # Make sure there are no `MISSING` values in the config
                for key, value in self.pydantic_model().model_dump().items():
                    if value is MISSING:
                        raise ValueError(
                            f"Config value for key '{key}' is `MISSING`.\n"
                            "Please provide a value for this key."
                        )

                self.__post_init__()

    # region MutableMapping implementation
    # These are under `if not TYPE_CHECKING` to prevent vscode from showing
    # all the MutableMapping methods in the editor
    if not TYPE_CHECKING:

        @property
        def _ll_dict(self):
            return self.model_dump()

        # we need to make sure every config class
        # is a MutableMapping[str, Any] so that it can be used
        # with lightning's hparams
        def __getitem__(self, key: str):
            # key can be of the format "a.b.c"
            # so we need to split it into a list of keys
            [first_key, *rest_keys] = key.split(".")
            value = self._ll_dict[first_key]

            for key in rest_keys:
                if isinstance(value, Mapping):
                    value = value[key]
                else:
                    value = getattr(value, key)

            return value

        def __setitem__(self, key: str, value: Any):
            # key can be of the format "a.b.c"
            # so we need to split it into a list of keys
            [first_key, *rest_keys] = key.split(".")
            if len(rest_keys) == 0:
                self._ll_dict[first_key] = value
                return

            # we need to traverse the keys until we reach the last key
            # and then set the value
            current_value = self._ll_dict[first_key]
            for key in rest_keys[:-1]:
                if isinstance(current_value, Mapping):
                    current_value = current_value[key]
                else:
                    current_value = getattr(current_value, key)

            # set the value
            if isinstance(current_value, MutableMapping):
                current_value[rest_keys[-1]] = value
            else:
                setattr(current_value, rest_keys[-1], value)

        def __delitem__(self, key: str):
            # this is unsupported for this class
            raise NotImplementedError

        @override
        def __iter__(self):
            return iter(self._ll_dict)

        def __len__(self):
            return len(self._ll_dict)

    # endregion

    @deprecated("No longer supported, use rich library instead.")
    def pprint(self):
        try:
            from rich import print as rprint
        except ImportError:
            warnings.warn(
                "rich is not installed, falling back to default print function"
            )
            print(self)
        else:
            rprint(self)


__all__ = [
    "MISSING",
    "Field",
    "ConfigBuilder",
    "TypedConfig",
]
